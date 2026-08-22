from __future__ import annotations

import argparse
import json
from pathlib import Path

from scholar_agent.evaluation.dev_plan_audit import audit_benchmark_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit benchmark artifact references")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-missing-external", action="store_true")
    args = parser.parse_args()
    report = audit_benchmark_artifacts(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    blocking = {"missing_tracked", "hash_mismatch", "unsafe_absolute"}
    if args.fail_on_missing_external:
        blocking.add("missing_external")
    return 1 if report["parse_error_count"] or any(
        item["status"] in blocking for item in report["records"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
