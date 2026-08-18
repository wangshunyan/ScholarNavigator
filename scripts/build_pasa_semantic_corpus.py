#!/usr/bin/env python3
"""Join a public arXiv title+abstract CSV to the PaSa title index.

The join is title-based because the lightweight public CSV does not carry
arXiv IDs. Ambiguous title matches are skipped rather than assigned
arbitrarily.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BuildReport:
    input_zip: str
    pasa_title_index: str
    output_path: str
    input_rows: int
    matched_rows: int
    ambiguous_matches: int
    output_rows: int
    output_abstract_rows: int
    output_sha256: str
    category_counts: dict[str, int]


def build_semantic_corpus(
    input_zip: Path,
    pasa_title_index: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
) -> BuildReport:
    title_index = json.loads(pasa_title_index.read_text(encoding="utf-8"))
    if not isinstance(title_index, dict):
        raise ValueError("PaSa title index must be a JSON object")
    title_to_ids: dict[str, list[str]] = {}
    for arxiv_id, title in title_index.items():
        title_to_ids.setdefault(_normalize_title(str(title)), []).append(
            str(arxiv_id)
        )

    rows: dict[str, dict[str, str]] = {}
    category_counts: Counter[str] = Counter()
    input_rows = matched_rows = ambiguous_matches = 0
    with zipfile.ZipFile(input_zip) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.endswith("/")
        ]
        if not csv_members:
            raise ValueError("semantic source zip contains no CSV file")
        with archive.open(csv_members[0]) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            required = {"titles", "summaries"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError("semantic source CSV lacks titles/summaries")
            for source_row in reader:
                input_rows += 1
                title = _clean_text(source_row.get("titles"))
                abstract = _clean_text(source_row.get("summaries"))
                if not title or not abstract:
                    continue
                matches = title_to_ids.get(_normalize_title(title), [])
                if not matches:
                    continue
                matched_rows += 1
                if len(matches) != 1:
                    ambiguous_matches += 1
                    continue
                arxiv_id = matches[0]
                if arxiv_id in rows and len(abstract) <= len(
                    rows[arxiv_id]["abstract"]
                ):
                    continue
                rows[arxiv_id] = {
                    "_id": arxiv_id,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                }
                for category in _parse_categories(source_row.get("terms")):
                    category_counts[category] += 1

    if not rows:
        raise ValueError("semantic corpus join produced no rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(rows[arxiv_id], ensure_ascii=False, sort_keys=True) + "\n"
        for arxiv_id in sorted(rows)
    )
    output_path.write_text(output_text, encoding="utf-8")
    digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    report = BuildReport(
        input_zip=str(input_zip.resolve()),
        pasa_title_index=str(pasa_title_index.resolve()),
        output_path=str(output_path.resolve()),
        input_rows=input_rows,
        matched_rows=matched_rows,
        ambiguous_matches=ambiguous_matches,
        output_rows=len(rows),
        output_abstract_rows=sum(bool(row["abstract"]) for row in rows.values()),
        output_sha256=digest,
        category_counts=dict(category_counts.most_common()),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def _normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").split())


def _parse_categories(value: Any) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    raw = raw.strip("[]")
    return [
        item.strip().strip("'\"")
        for item in raw.split(",")
        if item.strip().strip("'\"")
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PaSa title+abstract semantic corpus."
    )
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=Path("datasets/semantic/arxiv_data.csv.zip"),
    )
    parser.add_argument(
        "--pasa-title-index",
        type=Path,
        default=Path("datasets/pasa/paper_database/id2paper.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/semantic/pasa_papers_with_abstracts.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/benchmark_inputs/pasa_semantic_corpus_report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_semantic_corpus(
            args.input_zip,
            args.pasa_title_index,
            args.output,
            report_path=args.report,
        )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
