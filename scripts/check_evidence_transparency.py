#!/usr/bin/env python3
"""CLI for evidence_transparency_log_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from scholar_agent.evaluation.evidence_transparency_log import (
    EXIT_NO_PUBLIC_CHECKPOINT,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    TransparencyError,
    append_record,
    audit_current,
    build_checkpoint,
    build_current,
    canonical_json,
    consistency_proof,
    inclusion_proof,
    load_protocol,
    read_json,
    simulate_matrix,
    verify_checkpoint,
    verify_consistency_proof,
    verify_inclusion_proof,
    verify_log,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmark/evidence_transparency_log_v1_protocol.json"
DEFAULT_LOG = ROOT / "benchmark/evidence_transparency_log_v1_log.json"
DEFAULT_CHECKPOINT = (
    ROOT / "benchmark/evidence_transparency_log_v1_checkpoint.json"
)


class UsageError(RuntimeError):
    """Arguments do not match the public CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _result(status: str, exit_code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "protocol": "evidence_transparency_log_v1",
        "schema_version": "1",
        "status": status,
    }
    if reason is not None:
        value["reason_code"] = reason
    return value


def _emit(value: dict[str, Any], output: Path | None) -> None:
    if output is not None:
        write_json(output, value)
    sys.stdout.buffer.write(canonical_json(value))


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-log")
    build.add_argument("--output", type=Path)
    build.add_argument("--checkpoint-output", type=Path)

    append = sub.add_parser("append-dry-run")
    append.add_argument("--log", type=Path, required=True)
    append.add_argument("--record", type=Path, required=True)
    append.add_argument("--output", type=Path)

    verify = sub.add_parser("verify-log")
    verify.add_argument("--log", type=Path, default=DEFAULT_LOG)
    verify.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    verify.add_argument("--output", type=Path)

    inclusion = sub.add_parser("prove-inclusion")
    inclusion.add_argument("--log", type=Path, default=DEFAULT_LOG)
    inclusion.add_argument("--sequence", type=int, required=True)
    inclusion.add_argument("--output", type=Path)

    consistency = sub.add_parser("prove-consistency")
    consistency.add_argument("--old-log", type=Path, required=True)
    consistency.add_argument("--new-log", type=Path, required=True)
    consistency.add_argument("--output", type=Path)

    audit = sub.add_parser("audit-readiness")
    audit.add_argument("--log", type=Path, default=DEFAULT_LOG)
    audit.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    audit.add_argument("--output", type=Path)
    return parser


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    protocol = load_protocol(args.protocol)
    if args.command == "build-log":
        log, checkpoint = build_current(ROOT, protocol)
        if args.output is not None:
            write_json(args.output, log)
        if args.checkpoint_output is not None:
            write_json(args.checkpoint_output, checkpoint)
        report = verify_log(log)
        report["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
        report["checkpoint_status"] = checkpoint["status"]
        report["simulation"] = simulate_matrix()
        return report, EXIT_READY
    if args.command == "append-dry-run":
        log = append_record(read_json(args.log), read_json(args.record))
        report = verify_log(log)
        if args.output is not None:
            write_json(args.output, log)
        return report, EXIT_READY
    if args.command == "verify-log":
        log = read_json(args.log)
        report = verify_log(log)
        verify_checkpoint(read_json(args.checkpoint), log)
        report["checkpoint_verified"] = True
        if args.output is not None:
            write_json(args.output, report)
        return report, EXIT_READY
    if args.command == "prove-inclusion":
        proof = inclusion_proof(read_json(args.log), args.sequence)
        verify_inclusion_proof(proof)
        if args.output is not None:
            write_json(args.output, proof)
        return proof, EXIT_READY
    if args.command == "prove-consistency":
        proof = consistency_proof(
            read_json(args.old_log), read_json(args.new_log)
        )
        verify_consistency_proof(proof)
        if args.output is not None:
            write_json(args.output, proof)
        return proof, EXIT_READY
    if args.command == "audit-readiness":
        report = audit_current(
            ROOT,
            protocol,
            read_json(args.log),
            read_json(args.checkpoint),
        )
        if args.output is not None:
            write_json(args.output, report)
        return report, EXIT_NO_PUBLIC_CHECKPOINT
    raise TransparencyError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report, exit_code = _run(args)
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, "invalid_arguments")
        exit_code = EXIT_USAGE
    except TransparencyError as exc:
        report = _result(
            "log_or_release_consistency_violation",
            EXIT_VIOLATION,
            str(exc),
        )
        exit_code = EXIT_VIOLATION
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        report = _result(
            "log_or_release_consistency_violation",
            EXIT_VIOLATION,
            "input_or_protocol_invalid",
        )
        exit_code = EXIT_VIOLATION
    try:
        _emit(report, None)
    except (OSError, UnicodeError, ValueError, TypeError):
        fallback = _result(
            "log_or_release_consistency_violation",
            EXIT_VIOLATION,
            "output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
