#!/usr/bin/env python3
"""Build and verify formal_validation_preregistration_v1 offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_validation_preregistration import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_SEALED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    PreregistrationError,
    audit_readiness,
    build_seal,
    canonical_json,
    load_protocol,
    read_json,
    synthetic_amendment_matrix,
    verify_seal,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/formal_validation_preregistration_v1_protocol.json"
DEFAULT_SEAL = ROOT / "benchmark/formal_validation_preregistration_v1_seal.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def parser() -> argparse.ArgumentParser:
    value = Parser(description="Seal formal validation rules before external evidence.")
    value.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    value.add_argument("--repository-root", default=str(ROOT))
    commands = value.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--seal", default=str(DEFAULT_SEAL))
    simulation = commands.add_parser("simulate-amendment")
    simulation.add_argument("--output")
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--seal", default=str(DEFAULT_SEAL))
    audit.add_argument("--output")
    return value


def emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol))
        if args.command == "build":
            seal = build_seal(protocol, repository_root=root)
            write_json(Path(args.output), seal)
            emit(seal)
            return EXIT_SEALED
        if args.command == "simulate-amendment":
            report = synthetic_amendment_matrix()
            emit(report, args.output)
            return EXIT_SEALED
        seal = read_json(Path(args.seal))
        if args.command == "verify":
            report = verify_seal(seal, protocol, repository_root=root)
            emit(report)
            return EXIT_SEALED
        report = audit_readiness(protocol, seal, repository_root=root)
        emit(report, args.output)
        return EXIT_BLOCKED
    except UsageError:
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "usage_error",
                "exit_code": EXIT_USAGE,
            }
        )
        return EXIT_USAGE
    except (PreregistrationError, KeyError, OSError, UnicodeError, ValueError, TypeError):
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "preregistration_or_posthoc_violation",
                "exit_code": EXIT_VIOLATION,
                "error_code": "preregistration_input_or_integrity_invalid",
                "formal_validation_complete": False,
            }
        )
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
