#!/usr/bin/env python3
"""Validate evidence freshness and compute deterministic change impact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.validation_evidence_freshness import (  # noqa: E402
    EXIT_MISSING,
    EXIT_STALE,
    EXIT_USAGE,
    FreshnessBaselineMissing,
    FreshnessError,
    audit_release,
    canonical_json,
    git_impact,
    load_contract,
    verify_current,
    worktree_impact,
    write_json,
)


DEFAULT_CONTRACT = ROOT / "benchmark/validation_evidence_freshness_v1_contract.json"


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise FreshnessError("usage_error")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Check validation evidence freshness without recomputing quality.")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-current", "impact-worktree", "audit-release"):
        command = commands.add_parser(name)
        command.add_argument("--output")
    impact = commands.add_parser("impact")
    impact.add_argument("--from", dest="from_ref", required=True)
    impact.add_argument("--to", dest="to_ref", required=True)
    impact.add_argument("--output")
    return parser


def _emit(report: dict[str, object], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    sys.stdout.write(canonical_json(report))


def _error(status: str, exit_code: int, reason: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": "validation_evidence_freshness_v1",
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "formal_validation_complete": False,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        contract = load_contract(Path(args.contract), repository_root=root)
        if args.command == "verify-current":
            report = verify_current(contract, repository_root=root)
        elif args.command == "impact-worktree":
            report = worktree_impact(contract, repository_root=root)
        elif args.command == "impact":
            report = git_impact(contract, repository_root=root, from_ref=args.from_ref, to_ref=args.to_ref)
        else:
            report = audit_release(contract, repository_root=root)
        _emit(report, args.output)
        return int(report["exit_code"])
    except FreshnessBaselineMissing:
        report = _error("not_ready_missing_baseline", EXIT_MISSING, "required_baseline_unavailable")
    except FreshnessError as exc:
        if str(exc) == "usage_error":
            report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
        else:
            report = _error("stale_or_dependency_violation", EXIT_STALE, str(exc))
    except (OSError, ValueError, json.JSONDecodeError):
        report = _error("stale_or_dependency_violation", EXIT_STALE, "controlled_input_or_io_failure")
    _emit(report, None)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
