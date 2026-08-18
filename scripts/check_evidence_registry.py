#!/usr/bin/env python3
"""Generate or check the tracked experiment evidence registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.evidence_registry import (  # noqa: E402
    build_evidence_registry,
    check_evidence_registry,
    write_evidence_registry,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or gate tracked strategy evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("generate", "check"):
        command = commands.add_parser(command_name)
        command.add_argument(
            "--manifest", default="benchmark/evidence_registry_manifest.json"
        )
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        values = build_evidence_registry(manifest)
        write_evidence_registry(args.output, *values)
        print(
            json.dumps(
                {
                    "strategy_count": values[1]["strategy_count"],
                    "default_enabled": values[1]["default_enabled_strategy_ids"],
                    "evidence_status_counts": values[1]["evidence_status_counts"],
                    "execution": values[0]["execution"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report = check_evidence_registry(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
