#!/usr/bin/env python3
"""Run the pre-registered paired significance audit on frozen Replay artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholar_agent.evaluation.paired_significance import (
    run_paired_significance_audit,
    write_paired_significance_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit paired lexical-normalization effects without external I/O."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/lexical_normalization_significance_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows, statistics = run_paired_significance_audit(args.manifest)
    write_paired_significance_audit(
        args.output_dir,
        rows,
        statistics,
        args.manifest,
    )
    print(
        "paired_significance_complete "
        f"queries={len(rows)} "
        f"evaluable={statistics['pairing']['all_evaluable_query_count']} "
        f"strict={statistics['pairing']['strict_comparable_query_count']}"
    )


if __name__ == "__main__":
    main()
