#!/usr/bin/env python3
"""Select a deterministic bounded arXiv-ID probe from exported candidates."""

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

from scholar_agent.core.identity import normalize_arxiv_id  # noqa: E402


_SCHEMA_VERSION = "quality-evidence-probe-selection-v1"
_SELECTION_DOMAIN = "quality-evidence-probe-v1"


def select_probe_identifiers(
    candidate_identifiers: Path,
    *,
    limit: int = 20,
) -> tuple[list[str], dict[str, object]]:
    """Choose a stable hash-ranked subset without loading query or gold data."""

    if not 1 <= limit <= 20:
        raise ValueError("quality_evidence_probe_limit_out_of_range")
    try:
        raw = candidate_identifiers.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("quality_evidence_candidate_identifiers_unavailable") from exc
    identifiers = _canonical_arxiv_identifiers(text)
    if len(identifiers) < limit:
        raise ValueError("quality_evidence_probe_population_too_small")
    selected = sorted(
        identifiers,
        key=lambda identifier: (_selection_hash(identifier), identifier),
    )[:limit]
    return selected, {
        "schema_version": _SCHEMA_VERSION,
        "selection_method": "sha256_ranked_without_replacement",
        "selection_domain": _SELECTION_DOMAIN,
        "candidate_identifiers_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_identifier_count": len(identifiers),
        "selected_identifier_count": len(selected),
        "gold_or_query_content_loaded": False,
    }


def write_probe_selection(
    identifiers: Sequence[str],
    report: dict[str, object],
    *,
    identifiers_output: Path,
    report_output: Path,
) -> None:
    """Write compact new probe artifacts without overwriting historical outputs."""

    if identifiers_output.exists() or report_output.exists():
        raise ValueError("quality_evidence_probe_output_exists")
    identifiers_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    identifiers_output.write_text(
        "".join(f"{identifier}\n" for identifier in identifiers),
        encoding="utf-8",
    )
    complete_report = {
        **report,
        "selected_identifiers_sha256": _sha256_file(identifiers_output),
    }
    report_output.write_text(
        json.dumps(complete_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_arxiv_identifiers(text: str) -> list[str]:
    values = [line.strip() for line in text.splitlines() if line.strip()]
    if not values or values != sorted(set(values)):
        raise ValueError("quality_evidence_candidate_identifiers_not_canonical")
    for value in values:
        prefix, separator, raw_identifier = value.partition(":")
        normalized = normalize_arxiv_id(raw_identifier) if prefix == "arxiv" and separator else None
        if value != f"arxiv:{normalized}" or normalized is None:
            raise ValueError("quality_evidence_candidate_arxiv_identifier_required")
    return values


def _selection_hash(identifier: str) -> str:
    return hashlib.sha256(
        _SELECTION_DOMAIN.encode("utf-8") + b"\0" + identifier.encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-identifiers", required=True, type=Path)
    parser.add_argument("--identifiers-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--limit", default=20, type=int)
    args = parser.parse_args(argv)
    try:
        identifiers, report = select_probe_identifiers(
            args.candidate_identifiers,
            limit=args.limit,
        )
        write_probe_selection(
            identifiers,
            report,
            identifiers_output=args.identifiers_output,
            report_output=args.report_output,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
