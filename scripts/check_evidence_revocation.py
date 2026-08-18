#!/usr/bin/env python3
"""Audit evidence revocation controls without modifying historical evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.evidence_revocation import (  # noqa: E402
    DEFAULT_LEDGER,
    DEFAULT_PROTOCOL,
    EXIT_BLOCKED,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ActiveIncident,
    RevocationError,
    audit_current,
    canonical_json,
    load_current,
    load_protocol,
    propagation_report,
    read_json,
    simulate_incidents,
    verify_ledger,
)


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def parser() -> argparse.ArgumentParser:
    value = Parser()
    value.add_argument("--repository-root", default=str(ROOT))
    value.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    value.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-current")
    commands.add_parser("audit-readiness")
    simulation = commands.add_parser("simulate-incident")
    simulation.add_argument("--output")
    verify = commands.add_parser("verify-ledger")
    verify.add_argument("--input")
    impact = commands.add_parser("impact")
    impact.add_argument("--input")
    return value


def emit(value: dict[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol_path = _resolve(root, args.protocol)
        ledger_path = _resolve(root, args.ledger)
        if args.command in {"audit-current", "audit-readiness"}:
            report = audit_current(
                root, protocol_path=protocol_path, ledger_path=ledger_path
            )
        else:
            protocol, _, freshness, readiness = load_current(
                root, protocol_path=protocol_path, ledger_path=ledger_path
            )
            if args.command == "simulate-incident":
                report = simulate_incidents(protocol, freshness, readiness)
                if args.output:
                    Path(args.output).write_bytes(canonical_json(report))
            else:
                selected = read_json(Path(args.input)) if args.input else read_json(ledger_path)
                if args.command == "verify-ledger":
                    report = verify_ledger(
                        selected,
                        protocol,
                        freshness_contract=freshness,
                        repository_root=root,
                    )
                else:
                    report = propagation_report(
                        selected,
                        protocol,
                        freshness,
                        readiness,
                        repository_root=root,
                    )
        emit(report)
        return int(report["exit_code"])
    except UsageError:
        emit(
            {
                "exit_code": EXIT_USAGE,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "usage_error",
            }
        )
        return EXIT_USAGE
    except ActiveIncident as exc:
        emit(
            {
                "error_code": str(exc),
                "exit_code": EXIT_BLOCKED,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "active_incident_blocks_release",
            }
        )
        return EXIT_BLOCKED
    except RevocationError as exc:
        emit(
            {
                "error_code": str(exc),
                "exit_code": EXIT_VIOLATION,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "revocation_or_propagation_violation",
            }
        )
        return EXIT_VIOLATION
    except (OSError, UnicodeError, ValueError, TypeError):
        emit(
            {
                "error_code": "controlled_input_or_io_failure",
                "exit_code": EXIT_VIOLATION,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "revocation_or_propagation_violation",
            }
        )
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
