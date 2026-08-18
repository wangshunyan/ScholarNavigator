#!/usr/bin/env python3
"""Run the offline runtime_hermeticity_v1 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.runtime_hermeticity import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    GATE_NAME,
    SCHEMA_VERSION,
    SCORE_SCOPE,
    RuntimeHermeticityError,
    RuntimeHermeticityNotEligible,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_runtime_hermeticity_gate,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "runtime_hermeticity_v1_protocol.json"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RuntimeHermeticityError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description=(
            "Audit offline Replay runtime hermeticity in controlled subprocesses."
        )
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--output")
    check.add_argument(
        "--fault",
        choices=[
            "dotenv_read",
            "network_attempt",
            "forbidden_file_read",
            "forbidden_file_write",
            "cache_residue",
            "subprocess_attempt",
            "sensitive_environment_read",
            "sensitive_sentinel_echo",
            "hash_seed_semantic_drift",
            "timezone_semantic_drift",
            "cwd_semantic_drift",
            "home_semantic_drift",
        ],
    )
    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _error(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "score_scope": SCORE_SCOPE,
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
    except RuntimeHermeticityError:
        report = _error("usage_error", EXIT_USAGE_ERROR, "invalid_arguments")
        _emit(report, None)
        return EXIT_USAGE_ERROR
    output = getattr(args, "output", None)
    try:
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "audit-frozen":
            report = audit_frozen_baseline_eligibility(
                protocol, repository_root=root
            )
        else:
            report = run_runtime_hermeticity_gate(
                protocol,
                repository_root=root,
                fault=args.fault,
            )
        _emit(report, output)
        return int(report["exit_code"])
    except RuntimeHermeticityNotEligible:
        report = _error(
            "not_eligible",
            EXIT_NOT_ELIGIBLE,
            "required_offline_contract_unavailable",
        )
        _emit(report, output)
        return EXIT_NOT_ELIGIBLE
    except RuntimeHermeticityError:
        report = _error("usage_error", EXIT_USAGE_ERROR, "invalid_offline_input")
        _emit(report, output)
        return EXIT_USAGE_ERROR
    except (OSError, ValueError, json.JSONDecodeError):
        report = _error(
            "hermeticity_or_semantic_violation",
            EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION,
            "controlled_worker_failure",
        )
        _emit(report, output)
        return EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
