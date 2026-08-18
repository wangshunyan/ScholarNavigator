#!/usr/bin/env python3
"""Validate blind human labels and prepare deterministic adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.human_precision_adjudication import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_PENDING_OR_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    GATE_NAME,
    SCHEMA_VERSION,
    HumanPrecisionGateError,
    LabelIntegrityViolation,
    PackageNotEligible,
    invalid_report,
    load_protocol,
    not_eligible_report,
    run_human_precision_gate,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "human_precision_adjudication_v1_protocol.json"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise HumanPrecisionGateError("usage_error")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Validate blinded labels without exposing package internals."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--annotator-one")
    parser.add_argument("--annotator-two")
    parser.add_argument("--adjudication")
    parser.add_argument("--prior-resolved")
    parser.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _usage_report(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "state": "invalid",
        "exit_code": EXIT_USAGE_ERROR,
        "score_scope": "internal_non_official_human_precision",
        "reason": reason,
        "statistics": None,
        "violation_count": 0,
        "violations": [],
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "official_scorer_call_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except HumanPrecisionGateError:
        report = _usage_report("usage_error")
        _emit(report, None)
        return EXIT_USAGE_ERROR
    output = args.output
    try:
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        report = run_human_precision_gate(
            protocol,
            repository_root=root,
            annotator_one_path=(
                Path(args.annotator_one) if args.annotator_one else None
            ),
            annotator_two_path=(
                Path(args.annotator_two) if args.annotator_two else None
            ),
            adjudication_path=(
                Path(args.adjudication) if args.adjudication else None
            ),
            prior_resolved_path=(
                Path(args.prior_resolved) if args.prior_resolved else None
            ),
        )
        _emit(report, output)
        return int(report["exit_code"])
    except LabelIntegrityViolation as exc:
        report = invalid_report(exc)
        _emit(report, output)
        return int(report["exit_code"])
    except PackageNotEligible as exc:
        report = not_eligible_report(str(exc))
        _emit(report, output)
        return EXIT_PENDING_OR_NOT_ELIGIBLE
    except (HumanPrecisionGateError, OSError, ValueError):
        report = _usage_report("input_or_protocol_error")
        _emit(report, output)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
