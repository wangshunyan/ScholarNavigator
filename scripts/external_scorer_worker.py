#!/usr/bin/env python3
"""Pre-imported isolated worker for external_scorer_handoff_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.external_scorer_handoff import (  # noqa: E402
    SCHEMA_VERSION,
    execute_worker_request,
    stable_json_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    try:
        args = parser.parse_args(argv)
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if request.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("worker_schema_invalid")
        report = execute_worker_request(request)
        Path(args.response).write_bytes(stable_json_bytes(report))
        return 0 if report.get("status") == "completed" else 2
    except Exception:  # noqa: BLE001 - never expose scorer or environment detail
        try:
            Path(args.response).write_bytes(
                stable_json_bytes(
                    {
                        "reason": "worker_protocol_error",
                        "status": "violation",
                    }
                )
            )
        except Exception:  # noqa: BLE001
            return 4
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
