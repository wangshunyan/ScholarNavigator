#!/usr/bin/env python3
"""Run or verify the offline, gold-free source_fusion_ablation_v1 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.source_fusion_ablation import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    SCHEMA_VERSION,
    SourceFusionAblationError,
    SourceFusionNotEligible,
    run_source_fusion_ablation,
    verify_analysis,
    write_analysis,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "source_fusion_ablation_v1_protocol.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "source_fusion_ablation_v1_result"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("usage_error")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Audit frozen Record160 four-source fusion offline.")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", default=str(DEFAULT_OUTPUT))
    verify = commands.add_parser("verify")
    verify.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _failure(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_or_qrels_loaded": False,
            "quality_metric_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "run":
            cases, aggregate = run_source_fusion_ablation(
                Path(args.protocol), repository_root=Path(args.repository_root)
            )
            manifest = write_analysis(
                Path(args.output), cases, aggregate, Path(args.protocol)
            )
            report = {
                "schema_version": SCHEMA_VERSION,
                "analysis": CONTRACT_VERSION,
                "status": "completed",
                "exit_code": 0,
                "included_main_case_count": aggregate["closure"][
                    "included_main_case_count"
                ],
                "excluded_case_count": aggregate["closure"][
                    "excluded_no_successful_source_count"
                ],
                "reconstruction_exact_case_count": aggregate["closure"][
                    "reconstruction_exact_case_count"
                ],
                "manifest_sha256": manifest["files"]["aggregate"]["sha256"],
                "execution": aggregate["execution"],
            }
        else:
            report = verify_analysis(Path(args.output))
    except SourceFusionNotEligible as exc:
        report = _failure("not_eligible", EXIT_NOT_ELIGIBLE, str(exc))
    except SourceFusionAblationError as exc:
        report = _failure(
            "reconstruction_or_analysis_violation", EXIT_VIOLATION, str(exc)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        report = _failure("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
    _emit(report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
