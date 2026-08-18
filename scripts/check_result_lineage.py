#!/usr/bin/env python3
"""Run the offline result_lineage_v1 reconstruction gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.result_lineage import (  # noqa: E402
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    GATE_NAME,
    ResultLineageError,
    ResultLineageNotEligible,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_result_lineage_gate,
    write_json,
)
from scholar_agent.core.result_lineage import (  # noqa: E402
    RESULT_LINEAGE_CONTRACT,
    RESULT_LINEAGE_SCHEMA_VERSION,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "field_lineage_v1_protocol.json"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ResultLineageError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Validate field-level result lineage without network access."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--output")
    check.add_argument(
        "--fault",
        choices=["field_injection"],
        help="Deterministic lineage corruption used only to prove the gate.",
    )

    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _error_report(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": RESULT_LINEAGE_SCHEMA_VERSION,
        "contract": RESULT_LINEAGE_CONTRACT,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "score_scope": "lineage_only_not_quality_or_official_score",
        "reason": reason,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except ResultLineageError:
        report = _error_report("usage_error", EXIT_USAGE_ERROR, "invalid_arguments")
        _emit(report, None)
        return EXIT_USAGE_ERROR
    output = getattr(args, "output", None)
    try:
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "audit-frozen":
            report = audit_frozen_baseline_eligibility(
                protocol, repository_root=root
            )
        else:
            report = run_result_lineage_gate(
                protocol,
                repository_root=root,
                controlled_fault=args.fault,
            )
        _emit(report, output)
        return int(report["exit_code"])
    except ResultLineageNotEligible:
        report = _error_report(
            "not_eligible", EXIT_NOT_ELIGIBLE, "required_lineage_evidence_unavailable"
        )
        _emit(report, output)
        return EXIT_NOT_ELIGIBLE
    except (ResultLineageError, OSError, ValueError, json.JSONDecodeError):
        report = _error_report("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
        _emit(report, output)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
