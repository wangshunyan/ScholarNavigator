#!/usr/bin/env python3
"""Probe and verify formal_execution_host_attestation_v1 without network I/O."""

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

from scholar_agent.evaluation.formal_execution_host_attestation import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_QUALIFIED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    HostAttestationError,
    HostAttestationNotReady,
    audit_readiness,
    canonical_json,
    load_protocol,
    probe_host,
    simulate_profiles,
    validate_attestation,
    validate_attestation_freshness,
    write_json,
    _read_object,
)


DEFAULT_PROTOCOL = "benchmark/formal_execution_host_attestation_v1_protocol.json"
DEFAULT_ATTESTATION = (
    "benchmark/formal_execution_host_attestation_v1_evidence/"
    "current_attestation.json"
)


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

    probe = commands.add_parser("probe")
    probe.add_argument("--primary-root", default=".")
    probe.add_argument("--backup-root")
    probe.add_argument("--output")

    verify = commands.add_parser("verify-attestation")
    verify.add_argument("--attestation", default=DEFAULT_ATTESTATION)
    verify.add_argument("--check-current-host", action="store_true")
    verify.add_argument("--primary-root", default=".")
    verify.add_argument("--backup-root")

    simulate = commands.add_parser("simulate-profile")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--attestation", default=DEFAULT_ATTESTATION)
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
    if args.command == "probe":
        primary = _resolve(root, args.primary_root)
        backup = _resolve(root, args.backup_root) if args.backup_root else None
        value = probe_host(
            root,
            protocol,
            primary_root=primary,
            backup_root=backup,
        )
        if args.output:
            write_json(Path(args.output), value)
        return {
            **value,
            "exit_code": (
                EXIT_QUALIFIED
                if value["status"] == "host_qualified"
                else EXIT_NOT_READY
            ),
        }
    if args.command == "verify-attestation":
        value = _read_object(_resolve(root, args.attestation))
        options: dict[str, Any] = {}
        current_observation: dict[str, Any] | None = None
        if args.check_current_host:
            primary = _resolve(root, args.primary_root)
            backup = _resolve(root, args.backup_root) if args.backup_root else None
            if not primary.is_dir() or backup is None or not backup.is_dir():
                raise HostAttestationNotReady("current_target_observation_unavailable")
            current_observation = probe_host(
                root,
                protocol,
                primary_root=primary,
                backup_root=backup,
            )
        attestation = validate_attestation(
            value, protocol, require_qualified=False, **options
        )
        if current_observation is not None:
            attestation = validate_attestation_freshness(
                value, current_observation, protocol
            )
        return _status(
            attestation.status,
            EXIT_QUALIFIED
            if attestation.status == "host_qualified"
            else EXIT_NOT_READY,
            attestation_sha256=attestation.attestation_sha256,
            missing_observations=attestation.missing_observations,
            failed_capabilities=attestation.failed_capabilities,
        )
    if args.command == "simulate-profile":
        report = simulate_profiles(protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "audit-readiness":
        report = audit_readiness(
            root,
            protocol,
            attestation_path=_resolve(root, args.attestation),
        )
        if args.output:
            write_json(Path(args.output), report)
        return report
    raise UsageError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = _run(args)
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except HostAttestationNotReady as exc:
        report = _status(
            "not_ready_unverified_or_insufficient_host",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        HostAttestationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "host_capability_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, HostAttestationError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        _emit(report)
    except (OSError, UnicodeError, TypeError, ValueError):
        fallback = _status(
            "host_capability_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
