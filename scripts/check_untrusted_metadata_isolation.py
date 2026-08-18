#!/usr/bin/env python3
"""Run the offline untrusted_metadata_isolation_v1 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.core.untrusted_metadata import (  # noqa: E402
    CONTRACT_VERSION,
    SCHEMA_VERSION,
)
from scholar_agent.evaluation.untrusted_metadata_isolation import (  # noqa: E402
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    GATE_NAME,
    SCORE_SCOPE,
    UntrustedMetadataIsolationError,
    UntrustedMetadataIsolationNotEligible,
    audit_frozen_eligibility,
    load_protocol,
    run_gate,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UntrustedMetadataIsolationError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Audit untrusted academic metadata isolation.")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "benchmark" / "untrusted_metadata_isolation_v1_protocol.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("check-fixture")
    fixture.add_argument(
        "--fault", choices=["role_escape", "cross_query_pollution"]
    )
    commands.add_parser("audit-frozen")
    return parser


def _emit(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _error(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "score_scope": SCORE_SCOPE,
        "observation": {
            "real_llm_request_count": 0,
            "tool_call_count": 0,
            "network_request_count": 0,
            "subprocess_count": 0,
            "snapshot_write_count": 0,
            "file_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except UntrustedMetadataIsolationError:
        report = _error("usage_error", EXIT_USAGE_ERROR, "invalid_arguments")
        _emit(report)
        return EXIT_USAGE_ERROR
    try:
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "audit-frozen":
            report = audit_frozen_eligibility(protocol, repository_root=root)
        else:
            report = run_gate(protocol, repository_root=root, fault=args.fault)
    except UntrustedMetadataIsolationNotEligible:
        report = _error(
            "not_eligible", EXIT_NOT_ELIGIBLE, "required_isolation_evidence_unavailable"
        )
    except (OSError, ValueError, json.JSONDecodeError, UntrustedMetadataIsolationError):
        report = _error("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
    _emit(report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
