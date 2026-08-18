#!/usr/bin/env python3
"""Prepare, verify and installation-test a strict offline wheelhouse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.offline_wheelhouse_intake import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    WheelhouseError,
    WheelhouseNotReady,
    build_manifest,
    freeze_release_contract,
    install_test,
    synthetic_install_test,
    verify_manifest,
    write_json,
)
from scholar_agent.evaluation.release_candidate_reproducibility import (  # noqa: E402
    EXECUTION,
    canonical_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/offline_wheelhouse_intake_v1_protocol.json"
DEFAULT_LOCK = ROOT / "benchmark/python_dependency_lock_v1_manifest.json"
DEFAULT_MANIFEST = ROOT / "benchmark/offline_wheelhouse_intake_v1_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--wheelhouse")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-manifest", "verify", "install-test", "audit-release"):
        command = commands.add_parser(name)
        command.add_argument("--report")
        if name == "install-test":
            command.add_argument("--synthetic", action="store_true")
    return parser


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(status: str, code: int, reason: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": code,
        "reason": reason,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        protocol = _load(Path(args.protocol))
        lock = _load(Path(args.lock))
        wheelhouse = (
            Path(args.wheelhouse)
            if args.wheelhouse
            else ROOT / str(protocol["wheelhouse_default"])
        )
        manifest_path = Path(args.manifest)
        if args.command == "prepare-manifest":
            report = build_manifest(wheelhouse, lock, protocol)
            write_json(manifest_path, report)
            release_contract, python_closure = freeze_release_contract(
                ROOT, protocol, lock, report
            )
            write_json(ROOT / protocol["release_contract_output"], release_contract)
            write_json(
                ROOT / protocol["release_python_closure_output"], python_closure
            )
        elif args.command == "verify":
            report = verify_manifest(
                wheelhouse, lock, protocol, _load(manifest_path)
            )
        elif args.command == "install-test":
            report = (
                synthetic_install_test(protocol)
                if args.synthetic
                else install_test(wheelhouse, _load(manifest_path), ROOT)
            )
        else:
            verification = verify_manifest(
                wheelhouse, lock, protocol, _load(manifest_path)
            )
            installation = install_test(wheelhouse, _load(manifest_path), ROOT)
            qualified = (
                verification["exit_code"] == 0 and installation["exit_code"] == 0
            )
            report = {
                "schema_version": "1",
                "protocol": PROTOCOL,
                "status": (
                    "wheelhouse_qualified"
                    if qualified
                    else "not_ready_missing_required_wheels"
                    if verification["exit_code"] == EXIT_NOT_READY
                    and installation["exit_code"] == EXIT_NOT_READY
                    else "artifact_or_supply_chain_violation"
                ),
                "exit_code": (
                    0
                    if qualified
                    else EXIT_NOT_READY
                    if verification["exit_code"] == EXIT_NOT_READY
                    and installation["exit_code"] == EXIT_NOT_READY
                    else EXIT_VIOLATION
                ),
                "verification": verification,
                "installation": installation,
                "python_offline_install_not_qualified": not qualified,
                "real_wheelhouse_qualified": qualified,
                "formal_validation_complete": False,
                "execution": EXECUTION,
            }
        if args.report:
            write_json(Path(args.report), report)
    except WheelhouseNotReady as exc:
        report = _error("not_ready_missing_required_wheels", EXIT_NOT_READY, str(exc))
    except WheelhouseError as exc:
        report = _error("artifact_or_supply_chain_violation", EXIT_VIOLATION, str(exc))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        report = _error("artifact_or_supply_chain_violation", EXIT_VIOLATION, "controlled_input_or_io_failure")
    except SystemExit:
        report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
