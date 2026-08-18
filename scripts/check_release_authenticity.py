#!/usr/bin/env python3
"""Offline OpenSSH release-signing and trust-root gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from scholar_agent.evaluation.release_authenticity_signing import (
    ALGORITHM,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    NAMESPACE,
    PROTOCOL,
    SCHEMA_VERSION,
    AuthenticityError,
    AuthenticityNotReady,
    audit_current,
    build_envelope,
    canonical_json,
    empty_trust_root,
    generate_test_key,
    load_protocol,
    read_json,
    register_key,
    sha256_file,
    sign_envelope,
    verify_signature_package,
    verify_trust_root,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmark/release_authenticity_signing_v1_protocol.json"
DEFAULT_TRUST_ROOT = (
    ROOT / "benchmark/release_authenticity_signing_v1_trust_root.json"
)


class UsageError(RuntimeError):
    """Arguments do not match the CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _result(status: str, exit_code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }
    if reason is not None:
        value["reason_code"] = reason
    return value


def _emit(value: MappingLike) -> None:
    sys.stdout.buffer.write(canonical_json(value))


MappingLike = dict[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-test-key")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--key-identity", required=True)
    generate.add_argument("--trust-root-output", type=Path, required=True)

    sign = sub.add_parser("sign-dry-run")
    sign.add_argument("--artifact", type=Path, required=True)
    sign.add_argument("--artifact-type", required=True)
    sign.add_argument("--artifact-version", required=True)
    sign.add_argument("--code-commit", required=True)
    sign.add_argument("--key-identity", required=True)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--trust-root", type=Path, required=True)
    sign.add_argument("--transparency-root", required=True)
    sign.add_argument("--transparency-sequence", type=int, required=True)
    sign.add_argument("--readiness-status", required=True)
    sign.add_argument("--issuance-sequence", type=int, required=True)
    sign.add_argument("--output", type=Path, required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--signature-package", type=Path, required=True)
    verify.add_argument("--trust-root", type=Path, default=DEFAULT_TRUST_ROOT)

    trust = sub.add_parser("verify-trust-root")
    trust.add_argument("--trust-root", type=Path, default=DEFAULT_TRUST_ROOT)

    audit = sub.add_parser("audit-readiness")
    audit.add_argument("--trust-root", type=Path, default=DEFAULT_TRUST_ROOT)
    audit.add_argument("--output", type=Path)
    return parser


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    protocol = load_protocol(args.protocol)
    if args.command == "generate-test-key":
        _private, public_key, fingerprint = generate_test_key(
            args.output_dir,
            key_identity=args.key_identity,
        )
        trust_root = register_key(
            empty_trust_root(),
            key_identity=args.key_identity,
            public_key=public_key,
            test_only=True,
        )
        write_json(args.trust_root_output, trust_root)
        return (
            {
                "algorithm": ALGORITHM,
                "exit_code": EXIT_READY,
                "formal_validation_complete": False,
                "key_identity": args.key_identity,
                "namespace": NAMESPACE,
                "protocol": PROTOCOL,
                "public_key_fingerprint": fingerprint,
                "schema_version": SCHEMA_VERSION,
                "status": "test_key_generated",
                "test_only": True,
                "trust_root_sha256": trust_root["trust_root_sha256"],
            },
            EXIT_READY,
        )
    if args.command == "sign-dry-run":
        trust_root = read_json(args.trust_root)
        envelope = build_envelope(
            artifact_type=args.artifact_type,
            artifact_version=args.artifact_version,
            content_sha256=sha256_file(args.artifact),
            transparency_root=args.transparency_root,
            transparency_sequence=args.transparency_sequence,
            code_commit=args.code_commit,
            readiness_status=args.readiness_status,
            key_identity=args.key_identity,
            test_only=True,
        )
        package = sign_envelope(
            envelope,
            private_key=args.private_key,
            trust_root=trust_root,
            issuance_sequence=args.issuance_sequence,
        )
        write_json(args.output, package)
        return (
            {
                "artifact_type": args.artifact_type,
                "exit_code": EXIT_READY,
                "formal_validation_complete": False,
                "key_identity": args.key_identity,
                "package_sha256": package["package_sha256"],
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "signing_controls_ready",
                "test_only": True,
            },
            EXIT_READY,
        )
    if args.command == "verify":
        report = verify_signature_package(
            read_json(args.signature_package),
            trust_root=read_json(args.trust_root),
            artifact_sha256=sha256_file(args.artifact),
        )
        report.update(
            {
                "exit_code": EXIT_READY,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "status": "signing_controls_ready",
            }
        )
        return report, EXIT_READY
    if args.command == "verify-trust-root":
        report = verify_trust_root(read_json(args.trust_root))
        report.update(
            {
                "exit_code": EXIT_READY,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
            }
        )
        return report, EXIT_READY
    if args.command == "audit-readiness":
        report = audit_current(ROOT, protocol, read_json(args.trust_root))
        if args.output is not None:
            write_json(args.output, report)
        return report, int(report["exit_code"])
    raise AuthenticityError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report, exit_code = _run(args)
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, "invalid_arguments")
        exit_code = EXIT_USAGE
    except AuthenticityNotReady as exc:
        report = _result(
            "not_ready_missing_real_trust_anchor_or_signer",
            EXIT_NOT_READY,
            str(exc),
        )
        exit_code = EXIT_NOT_READY
    except AuthenticityError as exc:
        report = _result(
            "signature_or_trust_violation",
            EXIT_VIOLATION,
            str(exc),
        )
        exit_code = EXIT_VIOLATION
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        report = _result(
            "signature_or_trust_violation",
            EXIT_VIOLATION,
            "input_or_protocol_invalid",
        )
        exit_code = EXIT_VIOLATION
    try:
        _emit(report)
    except (OSError, UnicodeError, ValueError, TypeError):
        fallback = _result(
            "signature_or_trust_violation",
            EXIT_VIOLATION,
            "output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
