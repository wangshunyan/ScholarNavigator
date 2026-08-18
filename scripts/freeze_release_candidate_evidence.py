#!/usr/bin/env python3
"""Convert a temporary double-build report into stable non-binary evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.release_candidate_reproducibility import (  # noqa: E402
    summarize_double_build_report,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmark/release_candidate_reproducibility_v1_evidence/current.json",
    )
    args = parser.parse_args()
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    write_json(ROOT / args.output, summarize_double_build_report(report))
    print(json.dumps({"status": "release_evidence_frozen", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
