#!/usr/bin/env python3
"""Run or verify the frozen Record160 source reliability diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scholar_agent.evaluation.source_reliability_diagnostics import (
    CONTRACT_VERSION,
    EXIT_COMPLETED,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    SourceReliabilityError,
    SourceReliabilityNotEligible,
    run_source_reliability_diagnostics,
    verify_analysis,
    write_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmark/source_reliability_diagnostics_v1_protocol.json"
DEFAULT_OUTPUT = ROOT / "benchmark/source_reliability_diagnostics_v1_result"


def _emit(payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen Record160 source reliability offline."
    )
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE_ERROR if int(exc.code or 0) else EXIT_COMPLETED
    try:
        if args.command == "run":
            cases, aggregate = run_source_reliability_diagnostics(
                args.protocol,
                repository_root=args.repository_root,
            )
            manifest = write_analysis(args.output, cases, aggregate, args.protocol)
            _emit(
                {
                    "schema_version": "1",
                    "analysis": CONTRACT_VERSION,
                    "status": "completed",
                    "exit_code": EXIT_COMPLETED,
                    "included_main_case_count": aggregate["closure"][
                        "included_main_case_count"
                    ],
                    "excluded_case_count": aggregate["closure"][
                        "excluded_no_successful_source_count"
                    ],
                    "observed_snapshot_key_count": aggregate["closure"][
                        "observed_snapshot_key_count"
                    ],
                    "manifest_file_count": len(manifest["files"]),
                    "execution": aggregate["execution"],
                }
            )
            return EXIT_COMPLETED
        _emit(verify_analysis(args.output))
        return EXIT_COMPLETED
    except SourceReliabilityNotEligible as exc:
        _emit(
            {
                "schema_version": "1",
                "analysis": CONTRACT_VERSION,
                "status": "not_eligible",
                "exit_code": EXIT_NOT_ELIGIBLE,
                "reason": str(exc),
                "execution": {
                    "gold_or_qrels_loaded": False,
                    "network_request_count": 0,
                    "llm_request_count": 0,
                    "snapshot_write_count": 0,
                    "quality_metric_count": 0,
                },
            }
        )
        return EXIT_NOT_ELIGIBLE
    except SourceReliabilityError as exc:
        _emit(
            {
                "schema_version": "1",
                "analysis": CONTRACT_VERSION,
                "status": "violation",
                "exit_code": EXIT_VIOLATION,
                "reason": str(exc),
                "execution": {
                    "gold_or_qrels_loaded": False,
                    "network_request_count": 0,
                    "llm_request_count": 0,
                    "snapshot_write_count": 0,
                    "quality_metric_count": 0,
                },
            }
        )
        return EXIT_VIOLATION
    except (OSError, TypeError, ValueError, KeyError):
        _emit(
            {
                "schema_version": "1",
                "analysis": CONTRACT_VERSION,
                "status": "usage_error",
                "exit_code": EXIT_USAGE_ERROR,
                "reason": "invalid_offline_input",
                "execution": {
                    "gold_or_qrels_loaded": False,
                    "network_request_count": 0,
                    "llm_request_count": 0,
                    "snapshot_write_count": 0,
                    "quality_metric_count": 0,
                },
            }
        )
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
