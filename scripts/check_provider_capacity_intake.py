#!/usr/bin/env python3
"""Build and audit offline provider capacity declaration kits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.provider_capacity_declaration_intake import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    SOURCES,
    CapacityIntakeError,
    CapacityIntakeNotReady,
    audit_readiness,
    build_kit,
    canonical_json,
    declaration_from_kit,
    import_declarations,
    load_protocol,
    read_object,
    simulate_matrix,
    validate_declaration,
    verify_kit,
    write_json,
)


DEFAULT_PROTOCOL = (
    "benchmark/provider_capacity_declaration_intake_v1_protocol.json"
)


class UsageError(RuntimeError):
    """Arguments do not satisfy the frozen CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-kit")
    build.add_argument("--source", required=True, choices=SOURCES)
    build.add_argument("--challenge", required=True)
    build.add_argument("--issued-epoch", required=True, type=int)
    build.add_argument("--output", required=True)

    verify = commands.add_parser("verify-declaration")
    verify.add_argument("--kit", required=True)
    verify.add_argument("--declaration", required=True)
    verify.add_argument("--current-epoch", required=True, type=int)
    verify.add_argument("--output")

    ingest = commands.add_parser("import-dry-run")
    ingest.add_argument("--bundle-dir", required=True)
    ingest.add_argument("--ledger", required=True)
    ingest.add_argument("--current-epoch", required=True, type=int)
    ingest.add_argument("--output")

    matrix = commands.add_parser("simulate-matrix")
    matrix.add_argument("--output")

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
            source=args.source,
            challenge_id=args.challenge,
            issued_epoch=args.issued_epoch,
            output=_resolve(root, args.output),
        )
    if args.command == "verify-declaration":
        kit = _resolve(root, args.kit)
        verify_kit(kit, protocol, repository_root=root)
        contract, _template = declaration_from_kit(kit)
        declaration = validate_declaration(
            contract,
            read_object(_resolve(root, args.declaration)),
            current_epoch=args.current_epoch,
        )
        report = {
            **_status("capacity_declarations_qualified", EXIT_READY),
            "source": declaration["source"],
            "declaration_sha256": declaration["declaration_sha256"],
        }
    elif args.command == "import-dry-run":
        bundle = _resolve(root, args.bundle_dir)
        entries = {
            source: (
                bundle / f"{source}.zip",
                bundle / f"{source}.declaration.json",
            )
            for source in SOURCES
        }
        receipt = import_declarations(
            root,
            protocol,
            entries=entries,
            ledger_path=_resolve(root, args.ledger),
            current_epoch=args.current_epoch,
        )
        report = {
            **_status("capacity_declarations_qualified", EXIT_READY),
            "receipt_sha256": receipt["receipt_sha256"],
            "request_preservation": receipt["request_preservation"],
        }
    elif args.command == "simulate-matrix":
        report = simulate_matrix(root, protocol)
    elif args.command == "audit-readiness":
        report = audit_readiness(protocol)
    else:
        raise UsageError("unsupported_command")
    if getattr(args, "output", None):
        write_json(_resolve(root, args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except CapacityIntakeNotReady as exc:
        report = _status(
            "not_ready_missing_real_declarations",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        CapacityIntakeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "declaration_or_import_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, CapacityIntakeError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        fallback = _status(
            "declaration_or_import_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
