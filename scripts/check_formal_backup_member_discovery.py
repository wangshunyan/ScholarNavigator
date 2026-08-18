#!/usr/bin/env python3
"""Discover explicitly registered Full1000 backup-member candidates offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_backup_member_discovery import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    BackupMemberDiscoveryError,
    audit_readiness,
    canonical_json,
    discover,
    load_protocol,
    match_topologies,
    read_object,
    simulate_profiles,
    validate_candidate,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_backup_member_discovery_v1_protocol.json"


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

    discovery = commands.add_parser("discover")
    discovery.add_argument("--target", action="append", default=[])
    discovery.add_argument("--output")

    verify = commands.add_parser("verify-candidate")
    verify.add_argument("--candidate", required=True)

    match = commands.add_parser("match-topology")
    match.add_argument("--candidates", required=True)

    simulation = commands.add_parser("simulate-profiles")
    simulation.add_argument("--output")

    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--output")
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
    if reason_code:
        value["reason_code"] = reason_code.split(":", 1)[0]
    return value


def _targets(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path or alias in result:
            raise UsageError("invalid_target_binding")
        result[alias] = Path(raw_path)
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "discover":
        report = discover(protocol, _targets(args.target))
    elif args.command == "verify-candidate":
        candidate = validate_candidate(protocol, read_object(Path(args.candidate)))
        report = {
            **_status(
                (
                    "qualifying_candidates_discovered"
                    if candidate["candidate_complete"]
                    else "no_real_qualifying_candidates"
                ),
                EXIT_READY if candidate["candidate_complete"] else EXIT_NOT_READY,
            ),
            "candidate": candidate,
        }
    elif args.command == "match-topology":
        payload = read_object(Path(args.candidates))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise BackupMemberDiscoveryError("candidate_inventory_invalid")
        match = match_topologies(protocol, candidates)
        ready = any(row["status"] == "candidate_topology_match" for row in match["plans"])
        report = {
            **_status(
                (
                    "candidate_topology_match"
                    if ready
                    else "no_real_qualifying_candidates"
                ),
                EXIT_READY if ready else EXIT_NOT_READY,
            ),
            "topology_match": match,
        }
    elif args.command == "simulate-profiles":
        report = simulate_profiles(protocol)
    elif args.command == "audit-readiness":
        report = audit_readiness(protocol)
    else:
        raise UsageError("unsupported_command")
    if getattr(args, "output", None):
        write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError as exc:
        report = _status("usage_error", EXIT_USAGE, reason_code=str(exc))
    except (
        BackupMemberDiscoveryError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "discovery_or_identity_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, BackupMemberDiscoveryError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
