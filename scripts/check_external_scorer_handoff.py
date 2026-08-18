#!/usr/bin/env python3
"""Prepare and audit the external_scorer_handoff_v1 chain offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.external_scorer_handoff import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_BLOCKED,
    EXIT_USAGE,
    EXIT_VERIFIED,
    EXIT_VIOLATION,
    SCHEMA_VERSION,
    ExternalScorerBlocked,
    ExternalScorerError,
    audit_real_readiness,
    load_protocol,
    run_scorer,
    run_synthetic_matrix,
    stable_json_bytes,
    synthetic_handoff,
    verify_package,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "external_scorer_handoff_v1_protocol.json"


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ExternalScorerError(f"usage:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Audit an external scorer handoff without quality evaluation.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--synthetic", action="store_true")

    verify = commands.add_parser("verify-package")
    verify.add_argument("--package", required=True)

    run = commands.add_parser("run")
    run.add_argument("--package")
    run.add_argument("--handoff")
    run.add_argument("--output")
    run.add_argument("--synthetic-matrix", action="store_true")

    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None = None) -> None:
    if output:
        write_json(Path(output), report)
    sys.stdout.buffer.write(stable_json_bytes(report))


def _error(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "analysis": CONTRACT_VERSION,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "reason": reason,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "prepare":
            if not args.synthetic:
                report = _error(
                    "blocked_missing_official_scorer_or_complete_input",
                    EXIT_BLOCKED,
                    "record160_legacy_run_lacks_authoritative_run_manifest_and_commit_generation",
                )
                _emit(report)
                return EXIT_BLOCKED
            handoff = synthetic_handoff()
            write_json(Path(args.output), handoff)
            report = {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_VERIFIED,
                "handoff_sha256": handoff["content_sha256"],
                "official_score_generated": False,
                "schema_version": SCHEMA_VERSION,
                "status": "handoff_chain_verified",
            }
            _emit(report)
            return EXIT_VERIFIED
        if args.command == "verify-package":
            manifest = verify_package(Path(args.package), protocol)
            report = {
                "analysis": CONTRACT_VERSION,
                "entrypoint_sha256": manifest["entrypoint_sha256"],
                "exit_code": EXIT_VERIFIED,
                "official_score_generated": False,
                "schema_version": SCHEMA_VERSION,
                "status": "handoff_chain_verified",
            }
            _emit(report)
            return EXIT_VERIFIED
        if args.command == "run":
            if args.synthetic_matrix:
                report = run_synthetic_matrix(protocol, repository_root=root)
                _emit(report, args.output)
                return EXIT_VERIFIED
            if not args.package or not args.handoff:
                raise ExternalScorerError("usage:package_and_handoff_required")
            first = run_scorer(Path(args.package), Path(args.handoff), protocol, repository_root=root, run_ordinal=1)
            second = run_scorer(Path(args.package), Path(args.handoff), protocol, repository_root=root, run_ordinal=2)
            if first["output_bytes"] != second["output_bytes"]:
                raise ExternalScorerError("scorer_output_nondeterministic")
            report = {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_VERIFIED,
                "official_score_generated": False,
                "output_sha256": first["output_sha256"],
                "schema_version": SCHEMA_VERSION,
                "status": "handoff_chain_verified",
                "worker_audit": first["worker_audit"],
            }
            _emit(report, args.output)
            return EXIT_VERIFIED
        report = audit_real_readiness(protocol, repository_root=root)
        _emit(report, args.output)
        return EXIT_BLOCKED
    except ExternalScorerBlocked:
        report = _error(
            "blocked_missing_official_scorer_or_complete_input",
            EXIT_BLOCKED,
            "required_authoritative_input_unavailable",
        )
        _emit(report)
        return EXIT_BLOCKED
    except ExternalScorerError as exc:
        reason = str(exc)
        if reason.startswith("usage:"):
            report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
            _emit(report)
            return EXIT_USAGE
        report = _error("scorer_or_integrity_violation", EXIT_VIOLATION, reason)
        _emit(report)
        return EXIT_VIOLATION
    except (OSError, ValueError, json.JSONDecodeError):
        report = _error("scorer_or_integrity_violation", EXIT_VIOLATION, "controlled_io_or_schema_failure")
        _emit(report)
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
