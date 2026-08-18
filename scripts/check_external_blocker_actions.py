#!/usr/bin/env python3
"""Build and audit the deterministic external-blocker action pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_external_blocker_action_pack import (  # noqa: E402
    EXIT_MISSING, EXIT_READY, EXIT_USAGE, EXIT_VIOLATION, PROTOCOL,
    ExternalBlockerActionError, audit_chains, audit_readiness, build_pack,
    canonical_json, load_protocol, read_object, verify_pack, write_json,
)

DEFAULT_PROTOCOL = "benchmark/formal_external_blocker_action_pack_v1_protocol.json"
DEFAULT_PACK = "benchmark/formal_external_blocker_action_pack_v1_pack.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-pack")
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify-pack")
    verify.add_argument("--pack", default=DEFAULT_PACK)
    audit = commands.add_parser("audit-chains")
    audit.add_argument("--output")
    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--pack", default=DEFAULT_PACK)
    readiness.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _error(status: str, code: int, reason: str) -> dict[str, Any]:
    return {
        "execution": {"llm_request_count": 0,
                      "network_request_count": 0, "quality_metric_count": 0,
                      "snapshot_write_count": 0},
        "exit_code": code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "reason_code": reason.split(":", 1)[0],
        "schema_version": "1",
        "status": status,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "build-pack":
        # The baseline is closed after the generated pack is reviewed; live
        # verify/audit commands always perform the full freshness check.
        report = audit_chains(root, protocol, verify_freshness=False)
        pack = build_pack(protocol, report)
        write_json(Path(args.output), pack)
        return verify_pack(pack, protocol)
    if args.command == "verify-pack":
        pack = read_object(_resolve(root, args.pack))
        return verify_pack(pack, protocol)
    if args.command == "audit-chains":
        report = audit_chains(root, protocol)
    elif args.command == "audit-readiness":
        pack = read_object(_resolve(root, args.pack))
        report = audit_chains(root, protocol)
        if report["exit_code"] != EXIT_READY:
            return report
        report = audit_readiness(pack, protocol)
    else:
        raise UsageError("unsupported_command")
    if args.output:
        write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError:
        report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
    except (ExternalBlockerActionError, KeyError, OSError, TypeError, UnicodeError,
            ValueError, json.JSONDecodeError) as exc:
        report = _error("chain_or_pack_violation", EXIT_VIOLATION, str(exc))
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
