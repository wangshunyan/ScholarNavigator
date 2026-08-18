#!/usr/bin/env python3
"""Generate or validate deterministic offline run provenance manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.run_provenance import (  # noqa: E402
    EXIT_INTEGRITY_FAILURE,
    EXIT_USAGE_ERROR,
    RunProvenanceError,
    audit_legacy_profiles,
    build_run_manifest,
    validate_run_manifest,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate run_manifest_v1 without network access."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--spec", required=True)
    generate.add_argument("--output", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--output")

    legacy = commands.add_parser("audit-legacy")
    legacy.add_argument("--profile", required=True)
    legacy.add_argument("--output")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunProvenanceError("JSON root must be an object")
    return value


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.repository_root).resolve()
    try:
        if args.command == "generate":
            spec = _read_json(Path(args.spec))
            manifest = build_run_manifest(spec, repository_root=root)
            output = Path(args.output)
            write_json(output, manifest.model_dump(mode="json"))
            report = validate_run_manifest(output, repository_root=root)
            _emit(report, None)
            return int(report["exit_code"])
        if args.command == "validate":
            report = validate_run_manifest(
                Path(args.manifest), repository_root=root
            )
            _emit(report, args.output)
            return int(report["exit_code"])
        report = audit_legacy_profiles(Path(args.profile), repository_root=root)
        _emit(report, args.output)
        return int(report["exit_code"])
    except (OSError, json.JSONDecodeError, RunProvenanceError, ValueError) as exc:
        report = {
            "schema_version": "1",
            "gate": "run_provenance_gate_v1",
            "status": "invalid",
            "exit_code": EXIT_USAGE_ERROR,
            "violation_count": 1,
            "violations": [
                {
                    "kind": "usage_error",
                    "path": "$",
                    "expected": "valid offline provenance input",
                    "observed": type(exc).__name__,
                }
            ],
            "execution": {
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "gold_fields_accessed": False,
            },
        }
        _emit(report, getattr(args, "output", None))
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
