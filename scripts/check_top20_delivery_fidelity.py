#!/usr/bin/env python3
"""Run or verify the frozen Top-20 delivery fidelity gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.top20_delivery_fidelity import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    Top20DeliveryError,
    Top20DeliveryNotEligible,
    run_top20_delivery_fidelity,
    verify_analysis,
    write_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线验证 current_rules Top-20 跨出口交付保真。"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行冻结 Record160 交付审计")
    run.add_argument("--contract", required=True)
    run.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="验证既有交付审计产物")
    verify.add_argument("--output", required=True)
    return parser


def _emit(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "run":
            cases, results, aggregate = run_top20_delivery_fidelity(args.contract)
            manifest = write_analysis(
                args.output, cases, results, aggregate, args.contract
            )
            exit_code = int(aggregate["exit_code"])
            _emit(
                {
                    "analysis": CONTRACT_VERSION,
                    "exit_code": exit_code,
                    "final_result_count": len(results),
                    "manifest_file_count": len(manifest["files"]),
                    "query_count": sum(
                        item["analysis_status"] == "included_main_analysis"
                        for item in cases
                    ),
                    "status": str(aggregate["status"]),
                    "unsupported_or_ineligible_exports": aggregate[
                        "unsupported_or_ineligible_exports"
                    ],
                }
            )
            return exit_code
        if args.command == "verify":
            result = verify_analysis(args.output)
            _emit(result)
            return int(result["exit_code"])
        return EXIT_USAGE_ERROR
    except Top20DeliveryNotEligible as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_NOT_ELIGIBLE,
                "reason": str(exc),
                "status": "not_eligible",
            }
        )
        return EXIT_NOT_ELIGIBLE
    except Top20DeliveryError as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_VIOLATION,
                "reason": str(exc),
                "status": "delivery_or_roundtrip_violation",
            }
        )
        return EXIT_VIOLATION
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_USAGE_ERROR,
                "reason": type(exc).__name__,
                "status": "usage_error",
            }
        )
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
