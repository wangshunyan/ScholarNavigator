#!/usr/bin/env python3
"""Record/replay SciFact current-rules query-depth audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.audit_cross_source_gold import (  # noqa: E402
    _parent_throttle,
    _run_isolated,
)
from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.scifact_query_depth_audit import (  # noqa: E402
    DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
    SOURCES,
    DepthSnapshotStore,
    build_config,
    build_request_plan,
    fetch_page,
    load_oracle_rows,
    preflight_evidence,
    probe_request,
    read_results,
    record_missing,
    replay_audit,
    validate_preflight,
    write_replay_artifacts,
)
from scholar_agent.evaluation.datasets.beir_scifact import (  # noqa: E402
    load_beir_scifact_enriched,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SciFact current_rules 最大深度 200 检索曲线审计。"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("plan", "preflight", "record-missing", "replay"),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
    )
    return parser


def _runner(request: dict, timeout: float) -> dict:
    return _run_isolated(fetch_page, (request,), timeout)


def _load_plan(args: argparse.Namespace):  # noqa: ANN202 - compact CLI helper
    baseline = Path(args.baseline_run_dir).expanduser().resolve()
    config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    _validate_frozen_config(config)
    rows = read_results(baseline / "results.jsonl")
    requests, cases = build_request_plan(rows, config)
    return config, requests, cases


def _validate_frozen_config(config: dict) -> None:
    if config.get("dataset") != "beir_scifact":
        raise SystemExit("depth_audit_requires_beir_scifact")
    if config.get("query_planning_policy") != "current_rules":
        raise SystemExit("depth_audit_requires_current_rules")
    if config.get("ranking_policy") != "current_rules":
        raise SystemExit("depth_audit_requires_current_rules_ranking")
    if bool(config.get("enable_query_evolution")) or bool(config.get("enable_refchain")):
        raise SystemExit("depth_audit_forbidden_strategy_enabled")
    if (config.get("llm") or {}).get("llm_enabled"):
        raise SystemExit("depth_audit_llm_must_be_disabled")
    if int(config.get("top_k") or 0) != 20:
        raise SystemExit("depth_audit_requires_top20")


def _load_queries(args: argparse.Namespace, config: dict):
    queries = load_beir_scifact_enriched(
        args.dataset_path,
        crosswalk_path=args.crosswalk,
    )
    manifest = json.loads(Path(args.sample_manifest).read_text(encoding="utf-8"))
    expected = [str(item) for item in manifest["query_ids"]]
    observed = [query.query_id for query in queries]
    if observed != expected or observed != [str(item) for item in config["case_ids"]]:
        raise SystemExit("scifact_sample_manifest_mismatch")
    return queries


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, requests, cases = _load_plan(args)
    store = DepthSnapshotStore(args.snapshot_dir)
    preflight_dir = Path(args.preflight_dir).expanduser().resolve()

    if args.mode == "plan":
        unique_lists = {
            (item["source"], item["adapted_query"])
            for case in cases
            for item in case["lists"]
        }
        print(
            json.dumps(
                {
                    "case_count": len(cases),
                    "planned_list_count": sum(len(case["lists"]) for case in cases),
                    "unique_list_count": len(unique_lists),
                    "unique_page_key_count": len(requests),
                    "page_keys_by_source": {
                        source: sum(
                            request["source"] == source for request in requests.values()
                        )
                        for source in SOURCES
                    },
                    "gold_fields_in_request_plan": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "preflight":
        if not args.source:
            raise SystemExit("preflight_requires_source")
        source_requests = [
            request for request in requests.values() if request["source"] == args.source
        ]
        if not source_requests:
            raise SystemExit("preflight_source_has_no_request")
        load_project_env(REPO_ROOT)
        request = probe_request(source_requests[0])
        _parent_throttle(args.source)
        response = _runner(request, args.request_timeout)
        evidence = preflight_evidence(args.source, response)
        preflight_dir.mkdir(parents=True, exist_ok=True)
        (preflight_dir / f"{args.source}.json").write_text(
            json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "source": args.source,
                    "status": evidence["status"],
                    "error_type": evidence["error_type"],
                    "http_status": evidence["http_status"],
                    "diagnostics": evidence["diagnostics"],
                },
                sort_keys=True,
            )
        )
        return 0 if evidence["status"] == "success" else 2

    if args.mode == "record-missing":
        load_project_env(REPO_ROOT)
        selected_sources = (args.source,) if args.source else SOURCES
        preflights = {}
        for source in selected_sources:
            path = preflight_dir / f"{source}.json"
            if not path.is_file():
                raise SystemExit(f"depth_preflight_missing:{source}")
            evidence = json.loads(path.read_text(encoding="utf-8"))
            validate_preflight(evidence, source)
            preflights[source] = evidence
        # record_missing expects a complete source map for the selected request set.
        selected = [
            request
            for request in requests.values()
            if request["source"] in selected_sources
        ]
        counts = record_missing(
            selected,
            store,
            preflights,
            runner=_runner,
            throttle=_parent_throttle,
            wall_timeout_seconds=args.request_timeout,
            progress=lambda value: print(value, flush=True),
        )
        print(json.dumps({"source": args.source or "all", "counts": counts}, sort_keys=True))
        return 0 if counts.get("pending", 0) == 0 else 2

    if not args.output_dir:
        raise SystemExit("replay_requires_output_dir")
    queries = _load_queries(args, config)
    oracle_rows = load_oracle_rows(Path(args.oracle_dir) / "gold_audit.jsonl")
    records, aggregate = replay_audit(
        queries=queries,
        cases=cases,
        requests=requests,
        store=store,
        oracle_rows=oracle_rows,
        candidate_limit=int(config["budgets"]["max_candidate_papers"]),
    )
    audit_config = build_config(
        dataset_path=args.dataset_path,
        sample_manifest_path=args.sample_manifest,
        crosswalk_path=args.crosswalk,
        baseline_run_dir=args.baseline_run_dir,
        oracle_dir=args.oracle_dir,
        request_count=len(requests),
        list_count=sum(len(case["lists"]) for case in cases),
        candidate_limit=int(config["budgets"]["max_candidate_papers"]),
        request_wall_timeout_seconds=args.request_timeout,
    )
    hashes = write_replay_artifacts(
        args.output_dir,
        config=audit_config,
        records=records,
        aggregate=aggregate,
    )
    print(
        json.dumps(
            {
                "network_request_count": 0,
                "snapshot_write_count": 0,
                "depth_curve": aggregate["depth_curve"],
                "classification_counts": aggregate["classification_counts"],
                "artifact_hashes": hashes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
