#!/usr/bin/env python3
"""Fetch one explicitly licensed, allow-listed full-text source as JSON.

This is an opt-in utility. It performs no discovery, no authentication and no
fallback URL resolution.  Callers must explicitly provide the HTTPS host
allow-list and assert that the supplied license decision was verified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from scholar_agent.core.full_text_evidence import (
    DEFAULT_FULL_TEXT_TIMEOUT_SECONDS,
    DEFAULT_MAX_FULL_TEXT_BYTES,
    DEFAULT_MAX_PDF_PAGES,
    FullTextFetchResult,
    fetch_open_full_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Explicit HTTPS full-text URL")
    parser.add_argument("--license-id", required=True, help="Verified license identifier")
    parser.add_argument(
        "--license-verified",
        action="store_true",
        help="Acknowledge that the caller has independently verified the license",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        required=True,
        dest="allowed_hosts",
        help="Allowed HTTPS hostname; repeat for multiple hosts",
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_FULL_TEXT_TIMEOUT_SECONDS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FULL_TEXT_BYTES)
    parser.add_argument("--max-pdf-pages", type=int, default=DEFAULT_MAX_PDF_PAGES)
    parser.add_argument("--output", type=Path)
    return parser


def run_fetch(
    args: argparse.Namespace,
    *,
    opener: Callable[..., Any] | None = None,
) -> FullTextFetchResult:
    kwargs = {
        "source_url": args.url,
        "license_id": args.license_id,
        "license_verified": bool(args.license_verified),
        "allowed_hosts": {str(host).strip() for host in args.allowed_hosts if str(host).strip()},
        "timeout_seconds": args.timeout_seconds,
        "max_bytes": args.max_bytes,
        "max_pdf_pages": args.max_pdf_pages,
    }
    if opener is not None:
        kwargs["opener"] = opener
    return fetch_open_full_text(**kwargs)


def main(argv: list[str] | None = None, *, opener: Callable[..., Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fetch(args, opener=opener)
    rendered = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result.status == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())

