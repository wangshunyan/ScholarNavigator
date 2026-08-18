#!/usr/bin/env python3
"""Run or regression-check the frozen query-independence audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.query_independence_audit import (  # noqa: E402
    build_query_independence_audit,
    check_query_independence_regression,
    write_query_independence_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit deterministic query duplication and stratum independence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest",
            default="benchmark/autoscholar_query_independence_manifest.json",
        )
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = json.loads(
            Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8")
        )
        values = build_query_independence_audit(manifest)
        write_query_independence_audit(args.output, *values)
        summary = values[-1]
        print(
            json.dumps(
                {
                    "query_count": summary["query_count"],
                    "component_count": summary["component_count"],
                    "contaminated_query_count": summary["cross_stratum"][
                        "contaminated_query_count"
                    ],
                    "network_request_count": 0,
                    "llm_request_count": 0,
                    "snapshot_write_count": 0,
                    "output": str(Path(args.output).expanduser().resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report = check_query_independence_regression(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
