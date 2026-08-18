#!/usr/bin/env python3
"""One-shot offline freezer for release_candidate_reproducibility_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.release_candidate_reproducibility import (  # noqa: E402
    canonical_json,
    freeze_contract,
)


def main() -> int:
    spec = json.loads((ROOT / "benchmark/release_candidate_reproducibility_v1_spec.json").read_text(encoding="utf-8"))
    contract, locks = freeze_contract(ROOT, spec)
    (ROOT / "benchmark/release_candidate_reproducibility_v1_contract.json").write_bytes(canonical_json(contract))
    (ROOT / "benchmark/release_candidate_reproducibility_v1_python_lock.json").write_bytes(canonical_json(locks["python"]))
    print(json.dumps({"status": "contract_frozen", "source_file_count": len(contract["source_manifest"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
