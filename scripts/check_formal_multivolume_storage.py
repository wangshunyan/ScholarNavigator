#!/usr/bin/env python3
"""Audit Full1000 multi-volume topology and capacity without network I/O."""

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

from scholar_agent.evaluation.formal_multivolume_storage import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    MultiVolumeStorageError,
    MultiVolumeStorageNotReady,
    audit_readiness,
    build_launch_addendum,
    build_topology,
    canonical_json,
    load_profiles,
    load_protocol,
    read_object,
    simulate_run,
    verify_capacity,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_multivolume_storage_v1_protocol.json"


class UsageError(RuntimeError):
    """Arguments do not satisfy the public CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    topology = commands.add_parser("build-topology")
    topology.add_argument("--profiles", required=True)
    topology.add_argument("--output")
    topology.add_argument("--addendum-output")

    capacity = commands.add_parser("verify-capacity")
    capacity.add_argument("--profiles", required=True)
    capacity.add_argument("--topology", required=True)
    capacity.add_argument("--output")

    simulate = commands.add_parser("simulate-run")
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
    value: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    if reason_code is not None:
        value["reason_code"] = reason_code.split(":", 1)[0]
    return value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(
        _resolve(root, args.protocol), repository_root=root
    )
    if args.command == "build-topology":
        profiles = load_profiles(_resolve(root, args.profiles))
        topology = build_topology(root, protocol, profiles)
        addendum = build_launch_addendum(protocol)
        if args.output:
            write_json(Path(args.output), topology)
        if args.addendum_output:
            write_json(Path(args.addendum_output), addendum)
        return {
            **_status("multivolume_storage_ready", EXIT_READY),
            "topology_sha256": topology["topology_sha256"],
            "addendum_sha256": addendum["addendum_sha256"],
            "primary_volume_count": len(
                topology["primary_volume_identities"]
            ),
            "backup_volume_count": len(
                topology["backup_volume_identities"]
            ),
        }
    if args.command == "verify-capacity":
        profiles = load_profiles(_resolve(root, args.profiles))
        topology = read_object(_resolve(root, args.topology))
        report = verify_capacity(topology, profiles)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "simulate-run":
        report = simulate_run(root, protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
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
    except MultiVolumeStorageNotReady as exc:
        report = _status(
            "not_ready_missing_qualified_volumes",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        MultiVolumeStorageError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "topology_or_storage_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, MultiVolumeStorageError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        fallback = _status(
            "topology_or_storage_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
