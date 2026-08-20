#!/usr/bin/env python3
"""Serve a local Qwen instruction model through a loopback OpenAI-compatible API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.llm.local_provider import (  # noqa: E402
    LocalProviderError,
    TransformersLocalChatService,
    create_app,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model-path", required=True, type=Path)
    value.add_argument("--model-id", default="Qwen/Qwen3-4B")
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", default=18080, type=int)
    value.add_argument("--device", default="cuda:1")
    value.add_argument("--max-input-tokens", default=2048, type=_positive_int)
    return value


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        service = TransformersLocalChatService.load(
            model_path=str(args.model_path.expanduser().resolve()),
            device=args.device,
            max_input_tokens=args.max_input_tokens,
        )
    except LocalProviderError as exc:
        print(f"local_provider_start_failed:{exc}", file=sys.stderr)
        return 1

    import uvicorn

    uvicorn.run(create_app(service, model_id=args.model_id), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
