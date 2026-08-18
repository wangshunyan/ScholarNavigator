#!/usr/bin/env python3
"""Build and audit the official scorer offline intake chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.official_scorer_package_intake import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_QUALIFIED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    OfficialScorerIntakeError,
    OfficialScorerIntakeNotReady,
    audit_readiness,
    build_kit,
    canonical_json,
    conformance_dry_run,
    import_dry_run,
    load_protocol,
    simulate_matrix,
    verify_candidate_package,
)
from scholar_agent.evaluation.external_scorer_handoff import (  # noqa: E402
    ExternalScorerError,
)


DEFAULT_PROTOCOL = "benchmark/official_scorer_package_intake_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def parser() -> argparse.ArgumentParser:
    result = Parser(description=__doc__)
    result.add_argument("--repository-root", default=str(ROOT))
    result.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-kit")
    build.add_argument("--challenge", required=True)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify-package")
    verify.add_argument("--kit", required=True)
    verify.add_argument("--package", required=True)
    verify.add_argument("--synthetic", action="store_true")
    conform = commands.add_parser("conformance-dry-run")
    conform.add_argument("--kit")
    conform.add_argument("--package")
    conform.add_argument("--synthetic", action="store_true")
    conform.add_argument("--matrix", action="store_true")
    conform.add_argument("--output")
    ingest = commands.add_parser("import-dry-run")
    ingest.add_argument("--kit", required=True)
    ingest.add_argument("--package", required=True)
    ingest.add_argument("--ledger", required=True)
    ingest.add_argument("--synthetic", action="store_true")
    ingest.add_argument("--output")
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return result


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def status(name: str, code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": code,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": name,
    }
    if reason:
        value["reason_code"] = reason.split(":", 1)[0]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(resolve(root, args.protocol), repository_root=root)
    if args.command == "build-kit":
        return build_kit(
            root,
            protocol,
            challenge=args.challenge,
            output=resolve(root, args.output),
        )
    if args.command == "verify-package":
        manifest, _ = verify_candidate_package(
            resolve(root, args.kit),
            resolve(root, args.package),
            protocol,
            repository_root=root,
            allow_synthetic=args.synthetic,
        )
        return {
            **status("official_scorer_package_qualified", EXIT_QUALIFIED),
            "manifest_sha256": manifest["manifest_sha256"],
            "origin_status": manifest["origin_status"],
        }
    if args.command == "conformance-dry-run":
        if args.matrix:
            report = simulate_matrix(root, protocol)
        elif args.kit and args.package:
            report = conformance_dry_run(
                resolve(root, args.kit),
                resolve(root, args.package),
                protocol,
                repository_root=root,
                allow_synthetic=args.synthetic,
            )
        else:
            raise UsageError("invalid_arguments")
    elif args.command == "import-dry-run":
        report = import_dry_run(
            resolve(root, args.kit),
            resolve(root, args.package),
            resolve(root, args.ledger),
            protocol,
            repository_root=root,
            allow_synthetic=args.synthetic,
        )
    elif args.command == "audit-readiness":
        report = audit_readiness(protocol)
    elif args.command != "conformance-dry-run":
        raise UsageError("invalid_arguments")
    if getattr(args, "output", None):
        resolve(root, args.output).write_bytes(canonical_json(report))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except UsageError:
        report = status("usage_error", EXIT_USAGE, "invalid_arguments")
    except OfficialScorerIntakeNotReady as exc:
        report = status(
            "not_ready_missing_verified_official_package",
            EXIT_NOT_READY,
            str(exc),
        )
    except (
        OfficialScorerIntakeError,
        ExternalScorerError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        reason = (
            str(exc)
            if isinstance(exc, OfficialScorerIntakeError)
            else "controlled_schema_or_filesystem_violation"
        )
        report = status(
            "package_schema_or_sandbox_violation", EXIT_VIOLATION, reason
        )
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
