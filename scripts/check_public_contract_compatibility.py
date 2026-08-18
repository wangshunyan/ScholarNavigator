#!/usr/bin/env python3
"""Snapshot and verify public_contract_compatibility_v1 offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.public_contract_compatibility import (  # noqa: E402
    EXIT_BREAKING,
    EXIT_COMPATIBLE,
    EXIT_NOT_READY,
    EXIT_USAGE,
    PROTOCOL,
    SCHEMA_VERSION,
    ContractError,
    ContractNotReady,
    build_snapshot,
    canonical_json,
    compare_snapshots,
    load_json,
    load_protocol,
    validate_snapshot,
    verify_current,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
DEFAULT_BASELINE = ROOT / "benchmark/public_contract_compatibility_v1_baseline.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Fail-closed public contract compatibility gate.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--output")
    verify = commands.add_parser("verify-current")
    verify.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    compare = commands.add_parser("compare")
    compare.add_argument("--from", dest="from_path", required=True)
    compare.add_argument("--to", dest="to_path", required=True)
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _failure(status: str, code: int, reason: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "exit_code": code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }
    if reason:
        result["error_code"] = reason.split(":", 1)[0]
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol))
        if args.command == "snapshot":
            result = build_snapshot(protocol, repository_root=root)
            if args.output:
                output = Path(args.output)
                if output.exists():
                    raise ContractError("contract_baseline_overwrite_forbidden")
                write_json(output, result)
            _emit(result)
            return EXIT_COMPATIBLE
        if args.command == "compare":
            left = load_json(Path(args.from_path))
            right = load_json(Path(args.to_path))
            validate_snapshot(left)
            validate_snapshot(right)
            result = compare_snapshots(left, right)
            result["exit_code"] = EXIT_COMPATIBLE if result["classification"] == "compatible" else EXIT_BREAKING
            result["formal_validation_complete"] = False
            _emit(result)
            return int(result["exit_code"])
        baseline = load_json(Path(args.baseline))
        result = verify_current(protocol, baseline, repository_root=root)
        if args.command == "audit-readiness":
            result = dict(result)
            result["scope"] = "engineering_contract_governance_only"
        _emit(result)
        return EXIT_COMPATIBLE
    except UsageError:
        _emit(_failure("usage_error", EXIT_USAGE))
        return EXIT_USAGE
    except ContractNotReady as exc:
        _emit(_failure("not_ready_missing_contract_baseline", EXIT_NOT_READY, str(exc)))
        return EXIT_NOT_READY
    except (ContractError, OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        _emit(_failure("breaking_or_versioning_violation", EXIT_BREAKING, str(exc)))
        return EXIT_BREAKING


if __name__ == "__main__":
    raise SystemExit(main())
