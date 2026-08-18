#!/usr/bin/env python3
"""Record/replay SciFact exact-identifier cross-source indexability audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.scifact_source_index_audit import (  # noqa: E402
    DEFAULT_REQUEST_WALL_TIMEOUT_SECONDS,
    SOURCES,
    ExactLookupStore,
    build_config,
    build_request_plan,
    load_inputs,
    read_preflight,
    record_missing,
    replay_audit,
    run_exact_lookup_isolated,
    run_preflight,
    write_preflight,
    write_replay_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SciFact 四源精确标识可索引性 oracle 审计。"
    )
    parser.add_argument(
        "--mode", required=True, choices=("plan", "preflight", "record-missing", "replay")
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--external-run-dir", required=True)
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    queries = load_inputs(args.dataset_path, args.sample_manifest, args.crosswalk)
    requests, gold_plan = build_request_plan(queries)
    store = ExactLookupStore(args.snapshot_dir)
    preflight_dir = Path(args.preflight_dir).expanduser().resolve()

    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "gold_count": len(gold_plan),
                    "unique_gold_subject_count": len(
                        {item["audit_subject_id"] for item in gold_plan}
                    ),
                    "request_key_count": len(requests),
                    "by_source": {
                        source: {
                            "request_key_count": sum(
                                request.source == source for request in requests.values()
                            ),
                            "applicable_request_key_count": sum(
                                request.source == source and request.applicable
                                for request in requests.values()
                            ),
                        }
                        for source in SOURCES
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    if args.mode == "preflight":
        if not args.source:
            raise SystemExit("preflight_requires_source")
        load_project_env(REPO_ROOT)
        evidence = run_preflight(
            args.source,
            requests.values(),
            runner=run_exact_lookup_isolated,
            wall_timeout_seconds=args.request_timeout,
        )
        write_preflight(preflight_dir / f"{args.source}.json", evidence)
        print(
            json.dumps(
                {
                    "source": evidence.source,
                    "status": evidence.status,
                    "identifier_type": evidence.identifier_type,
                    "error_type": evidence.error_type,
                    "http_status": evidence.http_status,
                    "request_count": evidence.request_count,
                    "retry_count": evidence.retry_count,
                },
                sort_keys=True,
            )
        )
        return 0 if evidence.status != "failed" else 2

    if args.mode == "record-missing":
        load_project_env(REPO_ROOT)
        selected_sources = (args.source,) if args.source else SOURCES
        preflights = {
            source: read_preflight(preflight_dir / f"{source}.json", source)
            for source in selected_sources
        }
        counts = record_missing(
            requests.values(),
            store,
            preflights,
            source=args.source,
            runner=run_exact_lookup_isolated,
            wall_timeout_seconds=args.request_timeout,
        )
        print(json.dumps({"source": args.source or "all", "counts": counts}, sort_keys=True))
        return 0

    if not args.output_dir:
        raise SystemExit("replay_requires_output_dir")
    records, aggregate = replay_audit(
        queries=queries,
        requests=requests,
        gold_plan=gold_plan,
        store=store,
        external_run_dir=args.external_run_dir,
    )
    config = build_config(
        dataset_path=args.dataset_path,
        sample_manifest_path=args.sample_manifest,
        crosswalk_path=args.crosswalk,
        external_run_dir=args.external_run_dir,
        request_count=len(requests),
        gold_count=len(gold_plan),
        request_wall_timeout_seconds=args.request_timeout,
    )
    hashes = write_replay_artifacts(
        args.output_dir,
        config=config,
        records=records,
        aggregate=aggregate,
    )
    print(
        json.dumps(
            {
                "gold_count": aggregate["gold_count"],
                "source_pair_count": aggregate["source_pair_count"],
                "classification_counts": aggregate["classification_counts"],
                "joint_exact_coverage": aggregate["joint_exact_coverage"],
                "snapshot_status_counts": dict(
                    sorted(
                        Counter(
                            store.read(request).response.status
                            for request in requests.values()
                        ).items()
                    )
                ),
                "replay_http_request_count": 0,
                "artifact_hashes": hashes,
                "snapshot_directory_sha256": _directory_sha256(store.entries_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def _directory_sha256(path: Path) -> str:
    """Hash immutable snapshot files without copying them into Replay output."""

    import hashlib

    digest = hashlib.sha256()
    for item in sorted(path.glob("*.json")):
        relative = item.name.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
