#!/usr/bin/env python3
"""Audit a completed full neural-reranker contest run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_contest_qualification import _audit_reranker_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.run.expanduser().resolve()
    completed = list((root / ".run_commits" / "generations").glob("generation-*/RUN_COMPLETED"))
    reasons = [] if completed else ["run_not_completed"]
    audit = _audit_reranker_run(root, expected_rows=1000)
    reasons.extend(audit["reasons"])
    report = {
        "schema_version": "contest-full-reranker-audit-v1",
        "run_id": root.name,
        "status": "passed" if not reasons else "failed",
        "reasons": sorted(set(reasons)),
        "run_completed_generation_count": len(completed),
        "reranker": {**audit, "status": "passed" if not audit["reasons"] else "failed"},
        "internal_metric_scope": "not_official_competition_scorer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
