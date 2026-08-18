#!/usr/bin/env python3
"""Run or verify completion_bias_audit_v1 without retrieval or evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.completion_bias_audit import (  # noqa: E402
    ANALYSIS_NAME,
    EXIT_COMPLETED,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    CompletionBiasError,
    CompletionBiasNotEligible,
    canonical_json,
    run_completion_bias_audit,
    verify_analysis,
    write_analysis,
)


DEFAULT_PROTOCOL = "benchmark/completion_bias_audit_v1_protocol.json"
DEFAULT_OUTPUT = "benchmark/completion_bias_audit_v1_release"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Audit frozen Record completion bias offline.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        command.add_argument("--output", default=DEFAULT_OUTPUT)
        command.add_argument("--repository-root", default=str(ROOT))
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = root / args.protocol
        output = root / args.output
        if args.command == "run":
            queries, aggregate = run_completion_bias_audit(
                protocol, repository_root=root
            )
            bundle = write_analysis(output, queries, aggregate, protocol)
            report = verify_analysis(output, protocol, repository_root=root)
            report["aggregate_sha256"] = bundle["aggregate_sha256"]
            report["query_diagnostics_sha256"] = bundle[
                "query_diagnostics_sha256"
            ]
            _emit(report)
            return EXIT_COMPLETED
        _emit(verify_analysis(output, protocol, repository_root=root))
        return EXIT_COMPLETED
    except UsageError:
        _emit(
            {
                "analysis": ANALYSIS_NAME,
                "exit_code": EXIT_USAGE_ERROR,
                "protocol_version": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "usage_error",
            }
        )
        return EXIT_USAGE_ERROR
    except CompletionBiasNotEligible as exc:
        _emit(
            {
                "analysis": ANALYSIS_NAME,
                "error_code": str(exc),
                "exit_code": EXIT_NOT_ELIGIBLE,
                "protocol_version": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "not_eligible",
            }
        )
        return EXIT_NOT_ELIGIBLE
    except CompletionBiasError as exc:
        _emit(
            {
                "analysis": ANALYSIS_NAME,
                "error_code": str(exc),
                "exit_code": EXIT_VIOLATION,
                "protocol_version": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "identity_or_analysis_violation",
            }
        )
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
