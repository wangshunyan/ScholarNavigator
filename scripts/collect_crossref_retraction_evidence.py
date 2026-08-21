#!/usr/bin/env python3
"""Collect explicit Crossref retraction relations into a strict quality ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.core.quality_evidence_sources import (  # noqa: E402
    ArxivCrossrefRetractionCollection,
    CrossrefRetractionCollection,
    collect_arxiv_crossref_retraction_evidence,
    collect_crossref_retraction_evidence,
)


def load_paper_identifiers(path: Path) -> list[str]:
    """Read one canonical ``doi:`` identifier per non-comment line."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("crossref_identifier_input_unavailable") from exc
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_collection_outputs(
    collection: CrossrefRetractionCollection | ArxivCrossrefRetractionCollection,
    *,
    input_file_sha256: str,
    identifier_count: int,
    ledger_output: Path,
    report_output: Path,
) -> bool:
    """Write a ledger only when Crossref supplied an explicit risk relation."""

    if report_output.exists():
        raise ValueError("crossref_report_output_exists")
    if collection.evidence and ledger_output.exists():
        raise ValueError("crossref_ledger_output_exists")
    status = "evidence_written" if collection.evidence else "no_explicit_evidence"
    report = {
        "schema_version": "crossref-retraction-evidence-report-v1",
        "status": status,
        "source": (
            "arxiv_then_crossref"
            if isinstance(collection, ArxivCrossrefRetractionCollection)
            else "crossref"
        ),
        "input_file_sha256": input_file_sha256,
        "input_identifier_count": identifier_count,
        "flagged_evidence_count": len(collection.evidence),
        "outcome_counts": collection.outcome_counts(),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not collection.evidence:
        return False
    ledger_output.parent.mkdir(parents=True, exist_ok=True)
    ledger_output.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            + "\n"
            for item in collection.evidence
        ),
        encoding="utf-8",
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-identifiers",
        required=True,
        type=Path,
        help="UTF-8 text file with one canonical doi: or arxiv: identifier per line",
    )
    parser.add_argument(
        "--identifier-kind",
        choices=("doi", "arxiv"),
        required=True,
        help="arxiv mode resolves exact source metadata before optional Crossref lookup",
    )
    parser.add_argument("--ledger-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", default=10.0, type=float)
    args = parser.parse_args(argv)

    identifiers = load_paper_identifiers(args.paper_identifiers)
    collection = (
        collect_crossref_retraction_evidence(
            identifiers,
            timeout_seconds=args.timeout_seconds,
        )
        if args.identifier_kind == "doi"
        else collect_arxiv_crossref_retraction_evidence(
            identifiers,
            timeout_seconds=args.timeout_seconds,
        )
    )
    wrote_ledger = write_collection_outputs(
        collection,
        input_file_sha256=hashlib.sha256(args.paper_identifiers.read_bytes()).hexdigest(),
        identifier_count=len(set(identifiers)),
        ledger_output=args.ledger_output,
        report_output=args.report_output,
    )
    print(
        json.dumps(
            {
                "status": "evidence_written" if wrote_ledger else "no_explicit_evidence",
                "flagged_evidence_count": len(collection.evidence),
                "outcome_counts": collection.outcome_counts(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if wrote_ledger else 2


if __name__ == "__main__":
    raise SystemExit(main())
