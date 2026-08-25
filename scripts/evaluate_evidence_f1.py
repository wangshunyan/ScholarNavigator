#!/usr/bin/env python3
"""Score separately annotated full-text paragraph evidence.

This command is offline and intentionally unrelated to paper relevance gold or
qrels.  It exits with code 3 when the annotation files are not supplied; a
pending status is never represented as a zero score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.evaluation.evidence_f1 import (  # noqa: E402
    EvidenceF1Error,
    evaluate_evidence_f1,
    pending_evidence_f1,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.gold is None or args.predictions is None:
        result = pending_evidence_f1()
        _write(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 3
    try:
        gold = _load_jsonl(args.gold, "gold")
        predictions = _load_jsonl(args.predictions, "predictions")
        result = evaluate_evidence_f1(gold, predictions)
    except (EvidenceF1Error, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise EvidenceF1Error(f"{label}_row_not_object:{line_number}")
        rows.append(value)
    return rows


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
