#!/usr/bin/env python3
"""Run or check the frozen component-aware significance audit."""

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

from scholar_agent.evaluation.cluster_significance import (  # noqa: E402
    check_cluster_significance_regression,
    run_cluster_significance_audit,
    write_cluster_significance_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit lexical-normalization effects with frozen query components."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest",
            default="benchmark/lexical_normalization_cluster_significance_manifest.json",
        )
        command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        values = run_cluster_significance_audit(args.manifest)
        write_cluster_significance_audit(args.output, *values, args.manifest)
        statistics = values[-1]
        print(
            json.dumps(
                {
                    "analysis": statistics["analysis"],
                    "views": {
                        name: {
                            "queries": view["included_query_count"],
                            "components": view["component_count"],
                        }
                        for name, view in statistics["views"].items()
                    },
                    "execution": statistics["execution"],
                    "output": str(Path(args.output).expanduser().resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report = check_cluster_significance_regression(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
