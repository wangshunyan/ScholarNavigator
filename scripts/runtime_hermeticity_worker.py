#!/usr/bin/env python3
"""Private pre-imported worker for runtime_hermeticity_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Import the complete audited path before the worker enables content-I/O hooks.
from scholar_agent.evaluation.runtime_hermeticity import (  # noqa: E402
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    run_worker_request,
    stable_json_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    try:
        args = parser.parse_args(argv)
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if (
            request.get("contract") != CONTRACT_VERSION
            or request.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("worker_request_version_invalid")
        report = run_worker_request(request)
        Path(args.response).write_bytes(stable_json_bytes(report))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        # Never echo exception text, paths, request content, or environment state.
        report = {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTRACT_VERSION,
            "status": "worker_protocol_error",
            "reason": "invalid_worker_request",
        }
        try:
            Path(args.response).write_bytes(stable_json_bytes(report))
        except Exception:  # noqa: BLE001 - bootstrap cannot safely report further
            return 4
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
