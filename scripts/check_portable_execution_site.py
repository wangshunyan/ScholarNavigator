#!/usr/bin/env python3
"""Build and import portable Full1000 execution-site attestations offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.portable_execution_site_attestation import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    PortableSiteError,
    PortableSiteNotReady,
    audit_readiness,
    build_kit,
    canonical_json,
    import_attestation,
    load_protocol,
    simulate_sites,
    verify_kit,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/portable_execution_site_attestation_v1_protocol.json"


class UsageError(RuntimeError):
    """Arguments do not satisfy the public CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-kit")
    build.add_argument("--challenge", required=True)
    build.add_argument("--issued-epoch", required=True, type=int)
    build.add_argument("--output", required=True)

    verify = commands.add_parser("verify-kit")
    verify.add_argument("--kit", required=True)
    verify.add_argument("--output")

    ingest = commands.add_parser("import-attestation")
    ingest.add_argument("--kit", required=True)
    ingest.add_argument("--attestation", required=True)
    ingest.add_argument("--ledger", required=True)
    ingest.add_argument("--current-epoch", required=True, type=int)
    ingest.add_argument("--output")

    simulate = commands.add_parser("simulate-site")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(
    status: str, exit_code: int, *, reason_code: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    if reason_code:
        result["reason_code"] = reason_code.split(":", 1)[0]
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(
        _resolve(root, args.protocol), repository_root=root
    )
    if args.command == "build-kit":
        return build_kit(
            root,
            protocol,
            challenge_id=args.challenge,
            issued_epoch=args.issued_epoch,
            output=Path(args.output),
        )
    if args.command == "verify-kit":
        report = verify_kit(Path(args.kit), protocol, repository_root=root)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "import-attestation":
        receipt = import_attestation(
            root,
            protocol,
            kit_path=Path(args.kit),
            attestation_path=Path(args.attestation),
            ledger_path=Path(args.ledger),
            current_epoch=args.current_epoch,
        )
        if args.output:
            write_json(Path(args.output), receipt)
        return {
            **_status("execution_site_qualified", EXIT_READY),
            "receipt_sha256": receipt["receipt_sha256"],
            "attestation_sha256": receipt["attestation_sha256"],
            "challenge_id": receipt["challenge_id"],
        }
    if args.command == "simulate-site":
        report = simulate_sites(root, protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    raise UsageError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = _run(args)
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except PortableSiteNotReady as exc:
        report = _status(
            "not_ready_no_qualified_external_site",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        PortableSiteError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "attestation_or_import_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, PortableSiteError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        fallback = _status(
            "attestation_or_import_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
