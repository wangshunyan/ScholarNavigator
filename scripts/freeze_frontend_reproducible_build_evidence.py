#!/usr/bin/env python3
"""Freeze a successful temporary frontend qualification report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.frontend_reproducible_build import (  # noqa: E402
    freeze_release_contract,
    verify_evidence,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="benchmark/frontend_reproducible_build_v1_evidence/current.json",
    )
    args = parser.parse_args()
    protocol = json.loads(
        (ROOT / "benchmark/frontend_reproducible_build_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract, _ = freeze_release_contract(ROOT, protocol)
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    verified = verify_evidence(report, protocol, contract)
    if verified["exit_code"] != 0:
        raise SystemExit("frontend_evidence_not_qualified")
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if str(ROOT) in encoded:
        raise SystemExit("frontend_evidence_contains_absolute_path")
    write_json(ROOT / args.output, report)
    print(json.dumps({"output": args.output, "status": "evidence_frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
