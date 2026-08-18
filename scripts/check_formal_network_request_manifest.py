#!/usr/bin/env python3
"""Build and verify the offline Full1000 network-request intent manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.formal_network_request_manifest import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    NetworkRequestManifestError,
    NetworkRequestManifestNotReady,
    audit_readiness,
    audit_snapshots,
    build_launch_addendum,
    build_request_manifest,
    canonical_json,
    load_protocol,
    verify_bundle,
    write_bundle,
)


DEFAULT_PROTOCOL = "benchmark/formal_network_request_manifest_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    snapshots = commands.add_parser("audit-snapshots")
    snapshots.add_argument("--output")
    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(
    status: str, exit_code: int, *, reason_code: str | None = None
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }
    if reason_code:
        report["reason_code"] = reason_code.split(":", 1)[0]
    return report


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "build":
        intents, manifest = build_request_manifest(root, protocol)
        snapshots = audit_snapshots(root, protocol, intents)
        addendum = build_launch_addendum(protocol, manifest, snapshots)
        bundle = write_bundle(
            Path(args.output), intents, manifest, snapshots, addendum
        )
        return {
            **_status("request_manifest_ready_network_blocked", EXIT_READY),
            "bundle_sha256": bundle["bundle_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "logical_source_request_count": manifest[
                "logical_source_request_count"
            ],
            "http_attempt_upper": manifest["http_attempt_upper"],
        }
    if args.command == "verify":
        return verify_bundle(
            Path(args.bundle), protocol, repository_root=root
        )
    if args.command == "audit-snapshots":
        intents, _manifest = build_request_manifest(root, protocol)
        report = audit_snapshots(root, protocol, intents)
        result = {
            **_status("request_manifest_ready_network_blocked", EXIT_READY),
            "snapshot_audit": report,
        }
        if args.output:
            Path(args.output).write_bytes(canonical_json(result))
        return result
    if args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
        if args.output:
            Path(args.output).write_bytes(canonical_json(report))
        return report
    raise UsageError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = _run(args)
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except NetworkRequestManifestNotReady as exc:
        report = _status(
            "not_ready_missing_request_metadata",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        NetworkRequestManifestError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "request_identity_or_plan_violation",
            EXIT_VIOLATION,
            reason_code=str(exc),
        )
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
