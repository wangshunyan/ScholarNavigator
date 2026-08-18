#!/usr/bin/env python3
"""Plan, audit and merge deterministic offline Benchmark shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.crash_consistency import BenchmarkRunCommitStore  # noqa: E402
from scholar_agent.evaluation.run_provenance import (  # noqa: E402
    RunManifestV1,
    resolve_repo_path,
)
from scholar_agent.evaluation.sharded_execution import (  # noqa: E402
    EXIT_USAGE_ERROR,
    ShardedExecutionError,
    audit_frozen_eligibility,
    build_shard_plan,
    deterministic_fixture_report,
    load_gate_protocol,
    load_shard_plan,
    manifest_query_identities,
    validate_and_merge,
    validate_aggregate,
    write_shard_plan,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ShardedExecutionError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description="Validate shard_plan_v1 and merge committed offline shard runs."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "benchmark" / "sharded_execution_integrity_v1_protocol.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-plan")
    generate.add_argument("--run-manifest", required=True)
    generate.add_argument("--plan-id", required=True)
    generate.add_argument("--shard-count", type=int, required=True)
    generate.add_argument("--replay-input-sha256", required=True)
    generate.add_argument("--output", required=True)

    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", required=True)

    check = commands.add_parser("check")
    check.add_argument("--plan", required=True)
    check.add_argument("--attempts", required=True)
    check.add_argument("--monolithic-manifest", default=None)

    merge = commands.add_parser("merge")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--attempts", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--monolithic-manifest", default=None)

    aggregate = commands.add_parser("validate-aggregate")
    aggregate.add_argument("--plan", required=True)
    aggregate.add_argument("--attempts", required=True)
    aggregate.add_argument("--aggregate", required=True)

    fixture = commands.add_parser("check-fixture")
    fixture.add_argument("--shards", type=int, default=3)
    fixture.add_argument(
        "--fault",
        choices=["duplicate_query", "missing_query", "common_success_filter", "config_drift"],
    )
    fixture.add_argument("--incomplete-shard", type=int, default=None)
    fixture.add_argument("--retry-shard", type=int, default=None)

    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument(
        "--legacy-audit",
        default=str(ROOT / "benchmark" / "run_provenance_legacy_audit.json"),
    )
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _usage_report() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract": "sharded_execution_integrity_v1",
        "gate": "sharded_execution_integrity_gate",
        "status": "usage_error",
        "exit_code": EXIT_USAGE_ERROR,
        "score_scope": "partition_and_merge_only_not_quality_or_official_score",
        "reason": "invalid_offline_input",
        "observation": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        load_gate_protocol(Path(args.protocol))
        if args.command == "generate-plan":
            manifest_path = Path(args.run_manifest)
            manifest = RunManifestV1.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            query_ids = manifest_query_identities(manifest, root)
            state = BenchmarkRunCommitStore(
                resolve_repo_path(root, manifest.output_directory)
            ).load_latest()
            plan = build_shard_plan(
                plan_id=args.plan_id,
                monolithic_manifest=manifest,
                query_identities=query_ids,
                data_identity_sha256=manifest.dataset.identity_summary_sha256,
                replay_input_sha256=args.replay_input_sha256,
                shard_count=args.shard_count,
                generation_config=state.config,
            )
            write_shard_plan(Path(args.output), plan)
            report = {
                "schema_version": "1",
                "contract": "shard_plan_v1",
                "status": "passed",
                "exit_code": 0,
                "query_count": plan.queries.count,
                "shard_count": plan.shard_count,
                "assignment_algorithm": plan.assignment_algorithm,
                "score_scope": plan.score_scope,
            }
        elif args.command == "validate-plan":
            plan = load_shard_plan(Path(args.plan))
            report = {
                "schema_version": "1",
                "contract": "shard_plan_v1",
                "status": "passed",
                "exit_code": 0,
                "query_count": plan.queries.count,
                "shard_count": plan.shard_count,
                "assignment_algorithm": plan.assignment_algorithm,
                "score_scope": plan.score_scope,
            }
        elif args.command == "check":
            report = validate_and_merge(
                Path(args.plan),
                Path(args.attempts),
                repository_root=root,
                monolithic_manifest_path=(
                    Path(args.monolithic_manifest)
                    if args.monolithic_manifest
                    else None
                ),
            )
        elif args.command == "merge":
            report = validate_and_merge(
                Path(args.plan),
                Path(args.attempts),
                repository_root=root,
                output_path=Path(args.output),
                monolithic_manifest_path=(
                    Path(args.monolithic_manifest)
                    if args.monolithic_manifest
                    else None
                ),
            )
        elif args.command == "validate-aggregate":
            report = validate_aggregate(
                Path(args.aggregate),
                Path(args.plan),
                Path(args.attempts),
                repository_root=root,
            )
        elif args.command == "check-fixture":
            report = deterministic_fixture_report(
                shard_count=args.shards,
                incomplete_shard=args.incomplete_shard,
                retry_shard=args.retry_shard,
                controlled_fault=args.fault,
            )
        else:
            report = audit_frozen_eligibility(Path(args.legacy_audit))
        _emit(report)
        return int(report["exit_code"])
    except (ShardedExecutionError, OSError, ValueError, json.JSONDecodeError):
        report = _usage_report()
        _emit(report)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
