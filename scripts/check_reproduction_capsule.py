#!/usr/bin/env python3
"""Export, verify, or replay a reproduction_capsule_v1 offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.reproduction_capsule import (  # noqa: E402
    EXIT_INTEGRITY_OR_REPLAY_MISMATCH,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    CapsuleIntegrityError,
    CapsuleNotEligible,
    ReproductionCapsuleError,
    audit_frozen_baseline_eligibility,
    error_report,
    export_capsule,
    load_gate_protocol,
    replay_capsule,
    verify_capsule,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "reproduction_capsule_v1_protocol.json"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReproductionCapsuleError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Validate deterministic, data-only offline Replay capsules."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--source-root", required=True)
    export.add_argument("--capsule", required=True)
    export.add_argument("--output")

    verify = commands.add_parser("verify")
    verify.add_argument("--capsule", required=True)
    verify.add_argument("--output")

    replay = commands.add_parser("replay")
    replay.add_argument("--capsule", required=True)
    replay.add_argument("--output")
    replay.add_argument(
        "--fault",
        choices=["semantic_result_change"],
        help="Deterministic replay fault used only to prove exit code 2.",
    )

    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except ReproductionCapsuleError as exc:
        report = error_report(exc, EXIT_USAGE_ERROR)
        _emit(report, None)
        return EXIT_USAGE_ERROR
    root = Path(args.repository_root).resolve()
    output = getattr(args, "output", None)
    try:
        protocol = load_gate_protocol(Path(args.protocol), repository_root=root)
        if args.command == "export":
            report = export_capsule(
                Path(args.source_root),
                Path(args.capsule),
                host_repository_root=root,
            )
        elif args.command == "verify":
            report = verify_capsule(Path(args.capsule))
        elif args.command == "replay":
            report = replay_capsule(
                Path(args.capsule),
                host_repository_root=root,
                fault=args.fault,
            )
        else:
            report = audit_frozen_baseline_eligibility(
                protocol, repository_root=root
            )
        _emit(report, output)
        return int(report["exit_code"])
    except CapsuleNotEligible as exc:
        report = error_report(exc, EXIT_NOT_ELIGIBLE)
        _emit(report, output)
        return EXIT_NOT_ELIGIBLE
    except CapsuleIntegrityError as exc:
        report = error_report(exc, EXIT_INTEGRITY_OR_REPLAY_MISMATCH)
        _emit(report, output)
        return EXIT_INTEGRITY_OR_REPLAY_MISMATCH
    except (ReproductionCapsuleError, OSError, ValueError) as exc:
        error = (
            exc
            if isinstance(exc, ReproductionCapsuleError)
            else ReproductionCapsuleError(type(exc).__name__, stage="cli")
        )
        report = error_report(error, EXIT_USAGE_ERROR)
        _emit(report, output)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
