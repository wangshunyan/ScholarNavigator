#!/usr/bin/env python3
"""Validate Full1000 launch authorization without starting network execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.full1000_launch_control import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    LaunchControlError,
    LaunchControlNotReady,
    audit_readiness,
    build_authorization,
    build_preparation,
    canonical_json,
    load_protocol,
    simulate_operations,
    validate_authorization_context,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/full1000_launch_control_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Check the offline Full1000 launch-control seal.")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--state-dir", required=True)

    authorize = commands.add_parser("authorize-dry-run")
    authorize.add_argument("--state-dir", required=True)

    verify = commands.add_parser("verify-authorization")
    verify.add_argument("--state-dir", required=True)

    simulate = commands.add_parser("simulate-operations")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def _state_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_state(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LaunchControlNotReady("launch_state_missing") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchControlError("launch_state_malformed") from exc
    if not isinstance(value, dict):
        raise LaunchControlError("launch_state_not_object")
    return value


def _status(status: str, exit_code: int, **values: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        **values,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(root / args.protocol)
        if args.command == "simulate-operations":
            report = simulate_operations(root, protocol)
            _emit(report, args.output)
            return int(report["exit_code"])
        if args.command == "audit-readiness":
            report = audit_readiness(root, protocol)
            _emit(report, args.output)
            return int(report["exit_code"])

        state_dir = _state_path(root, args.state_dir)
        prepared_path = state_dir / "prepared.json"
        authorization_path = state_dir / "authorization.json"
        if args.command == "prepare":
            if state_dir.exists() and next(state_dir.iterdir(), None) is not None:
                raise LaunchControlError("launch_state_directory_not_empty")
            prepared = build_preparation(root, protocol)
            write_json(prepared_path, prepared)
            report = _status(
                "launch_controls_ready",
                EXIT_READY,
                state="prepared",
                preparation_sha256=prepared["preparation_sha256"],
                external_activation="blocked",
            )
        elif args.command == "authorize-dry-run":
            prepared = _load_state(prepared_path)
            authorization = build_authorization(prepared, protocol)
            validate_authorization_context(root, prepared, authorization, protocol)
            write_json(authorization_path, authorization)
            report = _status(
                "launch_controls_ready",
                EXIT_READY,
                state="authorized",
                authorization_sha256=authorization["authorization_sha256"],
                external_activation="blocked",
            )
        else:
            prepared = _load_state(prepared_path)
            authorization = _load_state(authorization_path)
            validate_authorization_context(root, prepared, authorization, protocol)
            report = _status(
                "launch_controls_ready",
                EXIT_READY,
                state="authorized",
                authorization_sha256=authorization["authorization_sha256"],
                external_activation="blocked",
            )
        _emit(report)
        return int(report["exit_code"])
    except UsageError:
        report = _status("usage_error", EXIT_USAGE)
    except LaunchControlNotReady as exc:
        report = _status(
            "external_activation_blocked",
            EXIT_BLOCKED,
            reason_code=str(exc),
        )
    except (LaunchControlError, KeyError, TypeError, ValueError, ValidationError) as exc:
        report = _status(
            "authorization_or_operation_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, LaunchControlError)
                else "controlled_schema_or_input_violation"
            ),
        )
    _emit(report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
