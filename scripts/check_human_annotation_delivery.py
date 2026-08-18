#!/usr/bin/env python3
"""Prepare and validate human_annotation_delivery_v1 offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.human_annotation_delivery import (  # noqa: E402
    EXIT_AWAITING,
    EXIT_USAGE,
    EXIT_VIOLATION,
    DeliveryError,
    DeliveryViolation,
    ingest,
    load_delivery_protocol,
    prepare_delivery,
    readiness,
    synthetic_dry_run,
    verify_delivery,
    write_json,
)

DEFAULT_PROTOCOL = ROOT / "benchmark/human_annotation_delivery_v1_protocol.json"
DEFAULT_PACKAGE = ROOT / "benchmark/human_annotation_delivery_v1_release"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "verify-package", "ingest", "dry-run", "audit-readiness"))
    p.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    p.add_argument("--repository-root", default=str(ROOT))
    p.add_argument("--package", default=str(DEFAULT_PACKAGE))
    p.add_argument("--annotator-a")
    p.add_argument("--annotator-b")
    p.add_argument("--output")
    p.add_argument("--replace-existing", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_delivery_protocol(Path(args.protocol), root)
        package = Path(args.package)
        if args.command == "prepare":
            report = prepare_delivery(protocol, repository_root=root, output=package, replace_existing=args.replace_existing)
        elif args.command == "verify-package":
            report = verify_delivery(protocol, package)
        elif args.command == "ingest":
            if not args.annotator_a or not args.annotator_b:
                raise DeliveryError("both_annotator_submissions_required")
            report = ingest(protocol, package_root=package, annotator_a=Path(args.annotator_a), annotator_b=Path(args.annotator_b), output=Path(args.output) if args.output else None)
            report.pop("recovered", None)
        elif args.command == "dry-run":
            report = synthetic_dry_run(protocol, repository_root=root)
        else:
            report = readiness(protocol, repository_root=root, package_root=package if package.exists() else None)
        if args.output and args.command != "ingest":
            write_json(Path(args.output), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return int(report["exit_code"])
    except DeliveryViolation as exc:
        report = {"schema_version": "1", "contract": "human_annotation_delivery_v1", "state": "invalid", "exit_code": EXIT_VIOLATION, "violation": {"code": exc.code, "path": exc.path}, "statistics": None}
    except (DeliveryError, OSError, ValueError, json.JSONDecodeError):
        report = {"schema_version": "1", "contract": "human_annotation_delivery_v1", "state": "usage_error", "exit_code": EXIT_USAGE, "statistics": None}
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
