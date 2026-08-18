#!/usr/bin/env python3
"""Freeze the b066 release input used by frontend_reproducible_build_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.frontend_reproducible_build import (  # noqa: E402
    freeze_release_contract,
    write_json,
)


def main() -> int:
    protocol = json.loads(
        (ROOT / "benchmark/frontend_reproducible_build_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract, _ = freeze_release_contract(ROOT, protocol)
    output = ROOT / "benchmark/frontend_reproducible_build_v1_release_contract.json"
    write_json(output, contract)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": "contract_frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
