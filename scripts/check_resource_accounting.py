#!/usr/bin/env python3
"""Validate resource_ledger_v1 without network, LLM, or Snapshot writes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.resource_accounting import (  # noqa: E402
    EXIT_USAGE_ERROR,
    ResourceAccountingError,
    ResourceLedgerV1,
    audit_evidence_registry,
    audit_frozen_eligibility,
    audit_shard_aggregate,
    deterministic_fixture_report,
    load_gate_protocol,
    validate_resource_ledger,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ResourceAccountingError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Run resource_accounting_integrity_v1 offline.")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "benchmark" / "resource_accounting_integrity_v1_protocol.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--ledger", required=True)

    fixture = commands.add_parser("check-fixture")
    fixture.add_argument(
        "--fault",
        choices=[
            "double_charge",
            "missing_call",
            "fake_cache_consumption",
            "negative_remaining",
            "over_budget",
            "post_cancel",
            "stale_attempt",
        ],
    )
    fixture.add_argument("--resume-shard", action="store_true")

    shards = commands.add_parser("check-shards")
    shards.add_argument("--aggregate", required=True)

    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument(
        "--legacy-audit",
        default=str(ROOT / "benchmark" / "run_provenance_legacy_audit.json"),
    )

    registry = commands.add_parser("audit-registry")
    registry.add_argument(
        "--registry",
        default=str(ROOT / "benchmark" / "evidence_registry_baseline" / "registry.json"),
    )
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _usage_report() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract": "resource_accounting_integrity_v1",
        "gate": "resource_accounting_integrity_gate",
        "status": "usage_error",
        "exit_code": EXIT_USAGE_ERROR,
        "reason": "invalid_offline_input",
        "observation": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
        "score_scope": "resource_accounting_only_not_quality_or_official_score",
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        load_gate_protocol(Path(args.protocol))
        if args.command == "check":
            ledger = ResourceLedgerV1.model_validate_json(
                Path(args.ledger).read_text(encoding="utf-8")
            )
            report = validate_resource_ledger(ledger)
        elif args.command == "check-fixture":
            report = deterministic_fixture_report(
                controlled_fault=args.fault,
                shard_resume=args.resume_shard,
            )
        elif args.command == "check-shards":
            report = audit_shard_aggregate(
                Path(args.aggregate), repository_root=root
            )
        elif args.command == "audit-frozen":
            report = audit_frozen_eligibility(Path(args.legacy_audit))
        else:
            report = audit_evidence_registry(Path(args.registry))
    except (OSError, ValueError, ResourceAccountingError, json.JSONDecodeError):
        report = _usage_report()
    _emit(report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
