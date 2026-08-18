#!/usr/bin/env python3
"""审计 current_rules 实际查询在固定深度前缀下的 Gold 召回曲线。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.audit_cross_source_gold import (  # noqa: E402
    _fetch_isolated,
    _hash,
    _matches,
    _parent_throttle,
    _source_outage_response,
    _validate_probe_evidence,
    _probe_evidence,
)
from scholar_agent.core.dedup import deduplicate_papers, normalize_title  # noqa: E402
from scholar_agent.evaluation.datasets import load_dataset  # noqa: E402
from scholar_agent.core.paper_schemas import Paper  # noqa: E402

SOURCES = ("arxiv", "openalex", "semantic_scholar", "pubmed")
PREFIXES = (20, 50, 100, 200)


def _rows(path: Path, case_ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["case_id"] in case_ids:
            rows.append(row)
    return rows


def _actual_queries(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    requests: dict[str, dict[str, Any]] = {}
    query_rows: list[dict[str, Any]] = []
    for row in rows:
        seen: dict[str, list[str]] = defaultdict(list)
        snapshots = row.get("stage_diagnostics", {}).get("snapshots", [])
        for snapshot in snapshots:
            if snapshot.get("stage") != "initial_retrieval":
                continue
            for call in snapshot.get("retrieval_calls", []):
                if call.get("logical_call_executed") is not True:
                    continue
                source = call.get("source")
                query = call.get("adapted_query")
                if source not in SOURCES or not query or query in seen[source]:
                    continue
                seen[source].append(query)
        per_source: dict[str, list[str]] = {}
        for source in SOURCES:
            per_source[source] = seen.get(source, [])
            for query in per_source[source]:
                request = {"source": source, "query": query, "limit": 200}
                request["key"] = _hash(request)
                requests[request["key"]] = request
        number = int(row["case_id"].rsplit("_", 1)[-1])
        query_rows.append({"case_id": row["case_id"], "split": "dev" if number < 10 else "val", "query": row["query"], "requests": per_source})
    return requests, query_rows


def _paper_identity(paper: dict[str, Any]) -> str:
    identifiers = paper.get("identifiers") or {}
    for field in ("doi", "arxiv_id", "openalex_id", "semantic_scholar_id", "pubmed_id"):
        value = identifiers.get(field)
        if value:
            return f"{field}:{str(value).strip().lower().rstrip('/')}"
    title = normalize_title(paper.get("title") or "")
    return f"title:{title}" if title else f"anonymous:{_hash(paper)}"


def _unique_prefix(papers: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    parsed = [Paper.model_validate(paper) for paper in papers[:limit]]
    return [paper.model_dump(mode="json") for paper in deduplicate_papers(parsed)]


def _match_gold(gold: dict[str, Any], papers: list[dict[str, Any]]) -> tuple[bool, int | None, bool]:
    for rank, paper in enumerate(papers, 1):
        stable, title = _matches(paper, gold)
        if stable or title:
            return True, rank, False
    uncertain = any(not (paper.get("title") or paper.get("identifiers")) for paper in papers)
    return False, None, uncertain


def _gold_rows() -> dict[str, list[dict[str, Any]]]:
    dataset = load_dataset("auto_scholar_query")
    return {
        query.query_id: [gold.model_dump(mode="json") for gold in query.gold_papers]
        for query in dataset
    }


def _classify(first_rank: int | None, unavailable: bool, uncertain: bool, incomplete: bool = False) -> str:
    if first_rank == 20:
        return "current_depth_hit"
    if first_rank in {50, 100, 200}:
        return "deeper_position_hit"
    if unavailable:
        return "source_unavailable"
    if incomplete:
        return "source_unavailable_or_incomplete"
    if uncertain:
        return "identity_match_uncertain"
    return "depth_200_miss"


def _update_rate_limit_streak(
    streak: int, failures: list[dict[str, Any]], response: dict[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    if response.get("failure_layer") == "http_429_rate_limit":
        return streak + 1, [*failures, response]
    return 0, []


def _write_snapshot_if_missing(path: Path, payload: dict[str, Any]) -> bool:
    if path.exists():
        return False
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return True


def _coverage_cell(gold_denominator: int, observed_matched_gold: int, incomplete_gold: int) -> dict[str, int]:
    return {
        "gold_denominator": gold_denominator,
        "evaluable_gold_denominator": gold_denominator - incomplete_gold,
        "observed_matched_gold": observed_matched_gold,
        "unavailable_or_incomplete_gold": incomplete_gold,
    }


def _record(args: argparse.Namespace, requests: dict[str, dict[str, Any]]) -> None:
    snapshot = Path(args.snapshot)
    run = Path(args.output)
    snapshot.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        evidence_path = run / f"probe_{source}.json"
        runtime_evidence_path = run / f"runtime_outage_{source}.json"
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        source_requests = [request for request in requests.values() if request["source"] == source]
        runtime_evidence: dict[str, Any] | None = None
        if runtime_evidence_path.exists():
            runtime_evidence = _validate_probe_evidence(json.loads(runtime_evidence_path.read_text()), source)
            for request in source_requests:
                path = snapshot / f"{request['key']}.json"
                if path.exists():
                    continue
                response = _source_outage_response(source, runtime_evidence)
                _write_snapshot_if_missing(path, {"request": request, "response": response})
            continue
        if evidence and evidence.get("status") != "success":
            validated = _validate_probe_evidence(evidence, source)
            for request in source_requests:
                path = snapshot / f"{request['key']}.json"
                if path.exists():
                    continue
                response = _source_outage_response(source, validated)
                _write_snapshot_if_missing(path, {"request": request, "response": response})
            continue
        consecutive_rate_limits = 0
        triggering_failures: list[dict[str, Any]] = []
        for request in source_requests:
            existing_path = snapshot / f"{request['key']}.json"
            if existing_path.exists():
                existing_response = json.loads(existing_path.read_text()).get("response", {})
                consecutive_rate_limits, triggering_failures = _update_rate_limit_streak(
                    consecutive_rate_limits, triggering_failures, existing_response
                )
        for index, request in enumerate(source_requests, 1):
            path = snapshot / f"{request['key']}.json"
            if path.exists():
                continue
            if consecutive_rate_limits >= 2:
                if runtime_evidence is None:
                    runtime_evidence = _probe_evidence(triggering_failures[-1], source)
                    attempted_keys = [
                        item["key"] for item in source_requests
                        if (snapshot / f"{item['key']}.json").exists()
                        and json.loads((snapshot / f"{item['key']}.json").read_text()).get("response", {}).get("status") in {"success", "failed"}
                    ]
                    not_sent_keys = [item["key"] for item in source_requests if item["key"] not in attempted_keys]
                    runtime_evidence["source_level_trigger"] = {
                        "threshold_consecutive_http_429": 2,
                        "triggering_failures": [
                            {"error_message": item.get("error_message"), "diagnostics": item.get("diagnostics", {})}
                            for item in triggering_failures[-2:]
                        ],
                        "attempted_request_keys": attempted_keys,
                        "not_sent_request_keys": not_sent_keys,
                        "attempted_count": len(attempted_keys),
                        "not_sent_count": len(not_sent_keys),
                    }
                    runtime_evidence_path.write_text(json.dumps(runtime_evidence, ensure_ascii=False, indent=2))
                response = _source_outage_response(source, runtime_evidence)
                _write_snapshot_if_missing(path, {"request": request, "response": response})
                continue
            _parent_throttle(source)
            response = _fetch_isolated(request, args.request_timeout)
            _write_snapshot_if_missing(path, {"request": request, "response": response})
            print(f"{source} [{index}/{len(source_requests)}] {response['status']}", flush=True)
            consecutive_rate_limits, triggering_failures = _update_rate_limit_streak(
                consecutive_rate_limits, triggering_failures, response
            )


def _probe(args: argparse.Namespace, requests: dict[str, dict[str, Any]]) -> None:
    source_requests = [request for request in requests.values() if request["source"] == args.source]
    if not source_requests:
        evidence = {"source": args.source, "status": "unavailable", "error_message": "no_current_rules_query", "error_type": "no_query", "diagnostics": {"request_count": 0}}
    else:
        request = source_requests[0]
        _parent_throttle(args.source)
        response = _fetch_isolated(request, args.request_timeout)
        evidence = _probe_evidence(response, args.source)
    output = Path(args.output) / f"probe_{args.source}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps(evidence, ensure_ascii=False))


def _replay(args: argparse.Namespace, requests: dict[str, dict[str, Any]], query_rows: list[dict[str, Any]]) -> None:
    snapshot = Path(args.snapshot)
    gold = _gold_rows()
    all_results = {}
    for key, request in requests.items():
        path = snapshot / f"{key}.json"
        if not path.exists():
            raise SystemExit(f"snapshot_missing:{key}")
        payload = json.loads(path.read_text())
        if payload.get("request") != request or _hash({k: request[k] for k in ("source", "query", "limit")}) != key:
            raise SystemExit(f"snapshot_request_mismatch:{key}")
        all_results[key] = payload
    detail: list[dict[str, Any]] = []
    classification = defaultdict(int)
    source_curve: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    overall_curve: dict[str, dict[str, dict[str, int]]] = {}
    first_hit_distribution = defaultdict(int)
    request_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in query_rows:
        split = row["split"]
        overall_hits: dict[str, set[int]] = {str(prefix): set() for prefix in PREFIXES}
        overall_incomplete: set[int] = set()
        gold_list = gold.get(row["case_id"], [])
        row_source_details: list[dict[str, Any]] = []
        for source in SOURCES:
            queries = row["requests"].get(source, [])
            responses = [all_results[_hash({"source": source, "query": query, "limit": 200})]["response"] for query in queries]
            for response in responses:
                request_status[source][response.get("status", "unknown")] += 1
            unavailable = not responses or all(response.get("status") in {"source_outage", "failed"} for response in responses)
            incomplete = not responses or any(response.get("status") != "success" for response in responses)
            papers = [paper for response in responses if response.get("status") == "success" for paper in response.get("papers", [])]
            prefix_unique: dict[int, list[dict[str, Any]]] = {}
            for prefix in PREFIXES:
                prefix_papers = []
                for response in responses:
                    prefix_papers.extend(response.get("papers", [])[:prefix] if response.get("status") == "success" else [])
                prefix_unique[prefix] = _unique_prefix(prefix_papers, len(prefix_papers))
            source_detail = {"source": source, "first_merged_rank": {}, "first_hit_depth": {}, "classification": {}, "prefixes": {}}
            for gold_index, gold_paper in enumerate(gold_list):
                first_merged_rank = None
                first_hit_depth = None
                uncertain = False
                prefix_hit: dict[str, bool] = {}
                for prefix in PREFIXES:
                    hit, rank, is_uncertain = _match_gold(gold_paper, prefix_unique[prefix])
                    prefix_hit[str(prefix)] = hit
                    uncertain = uncertain or is_uncertain
                    if hit and first_hit_depth is None:
                        first_hit_depth = prefix
                    if hit and first_merged_rank is None and prefix == 200:
                        first_merged_rank = rank
                    if hit:
                        overall_hits[str(prefix)].add(gold_index)
                    split_curve = source_curve.setdefault(source, {}).setdefault(split, {str(depth): {"gold_denominator": 0, "evaluable_gold_denominator": 0, "observed_matched_gold": 0, "unavailable_or_incomplete_gold": 0} for depth in PREFIXES})
                    split_curve[str(prefix)]["gold_denominator"] += 1
                    split_curve[str(prefix)]["evaluable_gold_denominator"] += int(not incomplete)
                    split_curve[str(prefix)]["observed_matched_gold"] += int(hit)
                    split_curve[str(prefix)]["unavailable_or_incomplete_gold"] += int(incomplete)
                if incomplete:
                    overall_incomplete.add(gold_index)
                category = _classify(first_hit_depth, unavailable, uncertain, incomplete and first_hit_depth is None)
                classification[category] += 1
                if first_hit_depth is not None:
                    first_hit_distribution[str(first_hit_depth)] += 1
                elif unavailable:
                    first_hit_distribution["unavailable"] += 1
                elif incomplete:
                    first_hit_distribution["incomplete"] += 1
                elif uncertain:
                    first_hit_distribution["uncertain"] += 1
                else:
                    first_hit_distribution["miss"] += 1
                identity = f"{gold_index}:{_hash(gold_paper)}"
                source_detail["first_merged_rank"][identity] = first_merged_rank
                source_detail["first_hit_depth"][identity] = first_hit_depth
                source_detail["classification"][identity] = category
                source_detail["prefixes"][identity] = prefix_hit
            if unavailable:
                status = "source_unavailable"
            elif any(response.get("status") != "success" for response in responses):
                status = "incomplete"
            else:
                status = "completed"
            row_source_details.append({"source": source, "query_count": len(queries), "status": status, "gold": source_detail})
        split_overall = overall_curve.setdefault(split, {str(depth): {"gold_denominator": 0, "evaluable_gold_denominator": 0, "observed_matched_gold": 0, "unavailable_or_incomplete_gold": 0} for depth in PREFIXES})
        case_overall: dict[str, dict[str, int]] = {}
        for prefix in PREFIXES:
            split_overall[str(prefix)]["gold_denominator"] += len(gold_list)
            split_overall[str(prefix)]["evaluable_gold_denominator"] += len(gold_list) - len(overall_incomplete)
            split_overall[str(prefix)]["observed_matched_gold"] += len(overall_hits[str(prefix)])
            split_overall[str(prefix)]["unavailable_or_incomplete_gold"] += len(overall_incomplete)
            case_overall[str(prefix)] = _coverage_cell(len(gold_list), len(overall_hits[str(prefix)]), len(overall_incomplete))
        increments = {}
        previous = 0
        for prefix in PREFIXES:
            increments[str(prefix)] = case_overall[str(prefix)]["observed_matched_gold"] - previous
            previous = case_overall[str(prefix)]["observed_matched_gold"]
        detail.append({"case_id": row["case_id"], "split": split, "overall_unique_gold": {"curve": case_overall, "increment": increments}, "sources": row_source_details})
    overall = {str(prefix): {"gold_denominator": 0, "evaluable_gold_denominator": 0, "observed_matched_gold": 0, "unavailable_or_incomplete_gold": 0} for prefix in PREFIXES}
    for split_curve in overall_curve.values():
        for prefix in PREFIXES:
            overall[str(prefix)]["observed_matched_gold"] += split_curve[str(prefix)]["observed_matched_gold"]
            overall[str(prefix)]["gold_denominator"] += split_curve[str(prefix)]["gold_denominator"]
            overall[str(prefix)]["evaluable_gold_denominator"] += split_curve[str(prefix)]["evaluable_gold_denominator"]
            overall[str(prefix)]["unavailable_or_incomplete_gold"] += split_curve[str(prefix)]["unavailable_or_incomplete_gold"]
    aggregate = {"depths": PREFIXES, "query_count": len(query_rows), "request_count": len(requests), "source_gold_pair_classification_counts": dict(classification), "first_hit_depth_distribution_source_gold_pairs": dict(first_hit_distribution), "source_curves": source_curve, "overall_unique_gold_curve": {**overall_curve, "overall": overall}, "overall_curve_completeness": "observed_lower_bound_incomplete_four_source_coverage", "request_status_by_source": {source: dict(statuses) for source, statuses in request_status.items()}, "replay_http_requests": 0, "source_query_counts": {source: sum(bool(row["requests"].get(source)) for row in query_rows) for source in SOURCES}}
    (Path(args.output) / "gold_depth_audit.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in detail) + "\n")
    (Path(args.output) / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "record-missing", "replay"), required=True)
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.mode == "probe" and not args.source:
        raise SystemExit("probe_requires_source")
    dataset = load_dataset("auto_scholar_query")
    case_ids = {query.query_id for query in dataset[0:10]} | {query.query_id for query in dataset[10:15]}
    dev_rows = _rows(ROOT / "outputs/benchmark_runs/source_ablation_d889_dev_all_replay/results.jsonl", {query.query_id for query in dataset[0:10]})
    val_rows = _rows(ROOT / "outputs/benchmark_runs/source_ablation_d889_val_all_replay/results.jsonl", {query.query_id for query in dataset[10:15]})
    requests, query_rows = _actual_queries(dev_rows + val_rows)
    if args.mode == "probe":
        _probe(args, requests)
    elif args.mode == "record-missing":
        _record(args, requests)
    else:
        _replay(args, requests, query_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
