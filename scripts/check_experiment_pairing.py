#!/usr/bin/env python3
"""Validate experiment treatment isolation and symmetric query pairing offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.experiment_pairing import (  # noqa: E402
    EXIT_USAGE_ERROR,
    ExperimentPairingError,
    audit_evidence_registry,
    audit_frozen_eligibility,
    deterministic_fixture_report,
    load_comparison_plan,
    load_gate_protocol,
    validate_pairing,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ExperimentPairingError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Check comparison_plan_v1 and paired committed runs without quality metrics."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "benchmark" / "experiment_pairing_integrity_v1_protocol.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("validate-plan")
    plan.add_argument("--plan", required=True)

    check = commands.add_parser("check")
    check.add_argument("--plan", required=True)
    check.add_argument("--baseline-manifest", required=True)
    check.add_argument("--candidate-manifest", required=True)

    fixture = commands.add_parser("check-fixture")
    fixture.add_argument(
        "--fault", choices=["hidden_treatment", "asymmetric_coverage"]
    )

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


def _emit(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _error_report(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract": "experiment_pairing_integrity_v1",
        "gate": "experiment_pairing_integrity_gate",
        "status": status,
        "exit_code": exit_code,
        "score_scope": "pairing_only_not_quality_or_official_score",
        "reason": reason,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        load_gate_protocol(Path(args.protocol), repository_root=root)
        if args.command == "validate-plan":
            plan = load_comparison_plan(Path(args.plan))
            report = {
                "schema_version": "1",
                "contract": "comparison_plan_v1",
                "status": "passed",
                "exit_code": 0,
                "query_count": plan.queries.count,
                "treatment_path_count": len(plan.allowed_treatment_changes),
                "predeclared_exclusion_count": len(plan.predeclared_exclusions),
                "score_scope": plan.score_scope,
            }
        elif args.command == "check":
            report = validate_pairing(
                Path(args.plan),
                Path(args.baseline_manifest),
                Path(args.candidate_manifest),
                repository_root=root,
            )
        elif args.command == "check-fixture":
            report = deterministic_fixture_report(controlled_fault=args.fault)
        elif args.command == "audit-frozen":
            report = audit_frozen_eligibility(Path(args.legacy_audit))
        else:
            report = audit_evidence_registry(Path(args.registry), repository_root=root)
        _emit(report)
        return int(report["exit_code"])
    except ExperimentPairingError:
        report = _error_report("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
        _emit(report)
        return EXIT_USAGE_ERROR
    except (OSError, ValueError, json.JSONDecodeError):
        report = _error_report("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
        _emit(report)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
