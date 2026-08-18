#!/usr/bin/env python3
"""Verify formal_provider_health_supervisor_v1 without network I/O."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_provider_health_supervisor import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_OBSERVED,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ProviderHealthError,
    ProviderHealthNotReady,
    audit_readiness,
    canonical_json,
    load_protocol,
    read_object,
    simulate_run,
    verify_resume_fixture,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_provider_health_supervisor_v1_protocol.json"


class UsageError(RuntimeError):
    """CLI arguments do not satisfy the public contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-policy")
    verify.add_argument("--output")
    simulate = commands.add_parser("simulate-run")
    simulate.add_argument("--output")
    resume = commands.add_parser("verify-resume")
    resume.add_argument("--evidence")
    resume.add_argument("--output")
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }


def _emit(report: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(report))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "verify-policy":
        report = _status(
            "supervisor_controls_ready",
            EXIT_READY,
            protocol_sha256=protocol["protocol_sha256"],
            source_count=len(protocol["population"]["sources"]),
            threshold_count=sum(
                len(values) for values in protocol["thresholds"].values()
            ),
        )
    elif args.command == "simulate-run":
        report = simulate_run(root, protocol)
    elif args.command == "verify-resume":
        evidence = read_object(_resolve(root, args.evidence)) if args.evidence else None
        report = verify_resume_fixture(root, protocol, evidence)
    elif args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
    else:
        raise UsageError("unsupported_command")
    if getattr(args, "output", None):
        write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except ProviderHealthNotReady as exc:
        report = _status(
            "external_provider_health_not_observed",
            EXIT_NOT_OBSERVED,
            reason_code=str(exc),
        )
    except (
        ProviderHealthError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "health_or_pause_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, ProviderHealthError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        _emit(report)
    except (OSError, TypeError, UnicodeError, ValueError):
        report = _status(
            "health_or_pause_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
