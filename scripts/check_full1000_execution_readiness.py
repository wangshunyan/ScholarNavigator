#!/usr/bin/env python3
"""Build and verify the immutable, offline Full1000 execution plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.full1000_execution_readiness import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    Full1000NotReady,
    Full1000ReadinessError,
    build_plan,
    canonical_json,
    dry_run,
    preflight,
    verify_plan,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/full1000_execution_readiness_v1_protocol.json"
DEFAULT_PLAN = "benchmark/full1000_execution_plan_v1.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Check Full1000 execution readiness without network access.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-plan", "verify-plan", "dry-run", "audit-readiness"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", default=str(ROOT))
        command.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        command.add_argument("--plan", default=DEFAULT_PLAN)
        command.add_argument("--output")
    return parser


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full1000NotReady("required_json_unavailable") from exc
    if not isinstance(value, dict):
        raise Full1000ReadinessError("required_json_not_object")
    return value


def _emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = _load_object(root / args.protocol)
        if args.command == "build-plan":
            plan = build_plan(root, protocol)
            write_json(root / args.plan, plan)
            report = verify_plan(root, protocol, plan)
            _emit(report, args.output)
            return int(report["exit_code"])
        plan = _load_object(root / args.plan)
        if args.command == "verify-plan":
            report = verify_plan(root, protocol, plan)
        elif args.command == "dry-run":
            verification = verify_plan(root, protocol, plan)
            report = verification if verification["exit_code"] else dry_run(plan)
        else:
            report = preflight(root, protocol, plan)
        _emit(report, args.output)
        return int(report["exit_code"])
    except UsageError:
        report = {
            "exit_code": EXIT_USAGE,
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "status": "usage_error",
        }
        _emit(report)
        return EXIT_USAGE
    except Full1000NotReady as exc:
        report = {
            "error_code": str(exc),
            "exit_code": EXIT_NOT_READY,
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "status": "not_ready_missing_required_input",
        }
        _emit(report)
        return EXIT_NOT_READY
    except Full1000ReadinessError as exc:
        report = {
            "error_code": str(exc),
            "exit_code": EXIT_VIOLATION,
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "status": "plan_or_preflight_violation",
        }
        _emit(report)
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
