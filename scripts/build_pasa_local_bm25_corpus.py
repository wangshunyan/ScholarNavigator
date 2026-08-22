#!/usr/bin/env python3
"""Convert PaSa/AutoScholar paper databases into local BM25 JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


Identity = Literal[
    "doi",
    "arxiv_id",
    "semantic_scholar_id",
    "s2orc_corpus_id",
    "openalex_id",
    "pubmed_id",
]
SourcePayload = Mapping[str, Any] | str
IDENTITIES: tuple[str, ...] = (
    "doi",
    "arxiv_id",
    "semantic_scholar_id",
    "s2orc_corpus_id",
    "openalex_id",
    "pubmed_id",
)
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "paper_title", "name"),
    "abstract": ("abstract", "summary", "text", "paper_abstract"),
    "doi": ("doi", "DOI", "external_ids.DOI", "externalids.DOI"),
    "arxiv_id": (
        "arxiv_id",
        "arxiv",
        "arxivId",
        "arxiv_id_v",
        "external_ids.ArXiv",
        "externalids.ArXiv",
    ),
    "semantic_scholar_id": (
        "semantic_scholar_id",
        "semanticScholarId",
        "paperId",
        "paper_id",
        "s2_id",
    ),
    "s2orc_corpus_id": (
        "s2orc_corpus_id",
        "s2orc_id",
        "corpus_id",
        "corpusid",
        "corpusId",
    ),
    "openalex_id": ("openalex_id", "openalex", "openalexId"),
    "pubmed_id": ("pubmed_id", "pmid", "pubmed", "pubmedId"),
    "authors": ("authors", "author", "authors_parsed", "author_names"),
    "year": ("year", "published", "publication_date", "update_date"),
    "venue": ("venue", "journal_ref", "journal-ref", "journal", "container_title"),
    "doi": ("doi", "DOI", "external_ids.DOI", "externalids.DOI"),
}


@dataclass
class ConversionReport:
    input_path: str
    zip_member: str | None
    output_path: str
    identity: str
    input_records: int
    output_records: int
    skipped_records: int
    duplicate_records: int
    skip_reasons: dict[str, int]
    output_sha256: str


def convert_corpus(
    input_path: Path,
    output_path: Path,
    *,
    identity: Identity = "arxiv_id",
    zip_member: str | None = None,
    report_path: Path | None = None,
    max_records: int | None = None,
) -> ConversionReport:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    skip_reasons: Counter[str] = Counter()
    input_count = 0
    duplicate_count = 0

    for source_id, payload in iter_source_records(input_path, zip_member=zip_member):
        input_count += 1
        if max_records is not None and input_count > max_records:
            break
        row, reason = normalize_record(source_id, payload, identity=identity)
        if row is None:
            skip_reasons[reason or "invalid_record"] += 1
            continue
        document_id = row["_id"]
        if document_id in seen:
            duplicate_count += 1
            continue
        seen.add(document_id)
        rows.append(row)

    if not rows:
        raise ValueError("conversion produced no local_bm25 rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    output_path.write_text(output_text, encoding="utf-8")
    output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    report = ConversionReport(
        input_path=str(input_path),
        zip_member=zip_member,
        output_path=str(output_path),
        identity=identity,
        input_records=input_count,
        output_records=len(rows),
        skipped_records=sum(skip_reasons.values()),
        duplicate_records=duplicate_count,
        skip_reasons=dict(sorted(skip_reasons.items())),
        output_sha256=output_sha,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def iter_source_records(
    input_path: Path,
    *,
    zip_member: str | None = None,
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if input_path.suffix.casefold() == ".zip":
        member, text = _read_zip_member(input_path, zip_member)
        yield from _iter_records_from_text(text, member)
        return
    text = input_path.read_text(encoding="utf-8")
    yield from _iter_records_from_text(text, input_path.name)


def normalize_record(
    source_id: str,
    payload: SourcePayload,
    *,
    identity: Identity,
) -> tuple[dict[str, str] | None, str | None]:
    if isinstance(payload, str):
        return _normalize_title_index_record(source_id, payload, identity=identity)

    title = _first_field(payload, FIELD_ALIASES["title"])
    if title is None:
        return None, "missing_title"
    identifiers = {
        name: _first_field(payload, FIELD_ALIASES[name])
        for name in IDENTITIES
    }
    document_id = identifiers.get(identity)
    if document_id is None:
        return None, f"missing_identity:{identity}"
    abstract = _first_field(payload, FIELD_ALIASES["abstract"]) or ""
    row: dict[str, Any] = {
        "_id": document_id,
        "title": title,
        "abstract": abstract,
    }
    for name, value in identifiers.items():
        if value:
            row[name] = value
    row.setdefault(identity, document_id)
    authors = _parse_list(_first_raw_field(payload, FIELD_ALIASES["authors"]))
    if authors:
        row["authors"] = authors
    year = _parse_year(_first_raw_field(payload, FIELD_ALIASES["year"]))
    if year is not None:
        row["year"] = year
    venue = _string_value(_first_raw_field(payload, FIELD_ALIASES["venue"]))
    if venue:
        row["venue"] = venue
    doi = _clean_doi(_first_raw_field(payload, FIELD_ALIASES["doi"]))
    if doi:
        row["doi"] = doi
    return row, None


def _normalize_title_index_record(
    source_id: str,
    title: str,
    *,
    identity: Identity,
) -> tuple[dict[str, str] | None, str | None]:
    """Handle PaSa's official ``arxiv_id -> title`` paper index."""

    normalized_title = _string_value(title)
    if normalized_title is None:
        return None, "missing_title"
    if identity != "arxiv_id":
        return None, f"missing_identity:{identity}"
    normalized_id = _string_value(source_id)
    if normalized_id is None:
        return None, "missing_identity:arxiv_id"
    return (
        {
            "_id": normalized_id,
            "title": normalized_title,
            "abstract": "",
            "arxiv_id": normalized_id,
        },
        None,
    )


def _read_zip_member(
    input_path: Path,
    zip_member: str | None,
) -> tuple[str, str]:
    with zipfile.ZipFile(input_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith((".json", ".jsonl"))
        ]
        if zip_member is None:
            preferred = [
                name
                for name in names
                if Path(name).name in {"id2paper.json", "id2paper.jsonl"}
            ]
            candidates = preferred or names
            if len(candidates) != 1:
                raise ValueError(
                    "zip contains multiple JSON members; pass --zip-member"
                )
            zip_member = candidates[0]
        with archive.open(zip_member) as handle:
            return zip_member, handle.read().decode("utf-8")


def _iter_records_from_text(
    text: str,
    name: str,
) -> Iterator[tuple[str, SourcePayload]]:
    if name.casefold().endswith(".jsonl"):
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                yield str(item.get("_id") or item.get("id") or line_number), item
        return

    payload = json.loads(text)
    if isinstance(payload, Mapping) and isinstance(payload.get("id2paper"), Mapping):
        payload = payload["id2paper"]
    if isinstance(payload, Mapping) and all(
        isinstance(value, Mapping) for value in payload.values()
    ):
        for key, value in payload.items():
            yield str(key), value
        return
    if isinstance(payload, Mapping) and all(
        isinstance(value, str) for value in payload.values()
    ):
        for key, value in payload.items():
            yield str(key), value
        return
    if isinstance(payload, list):
        for index, value in enumerate(payload, start=1):
            if isinstance(value, Mapping):
                yield str(value.get("_id") or value.get("id") or index), value
        return
    raise ValueError(f"unsupported paper database structure: {name}")


def _first_field(payload: Mapping[str, Any], paths: tuple[str, ...]) -> str | None:
    value = _first_raw_field(payload, paths)
    return _string_value(value)


def _first_raw_field(payload: Mapping[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = _lookup_path(payload, path)
        if value not in (None, "", [], {}):
            return value
    return None


def _lookup_path(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        if part in value:
            value = value[part]
            continue
        lowered = part.casefold()
        matching_key = next(
            (key for key in value if str(key).casefold() == lowered),
            None,
        )
        if matching_key is None:
            return None
        value = value[matching_key]
    return value


def _string_value(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            normalized = _string_value(item)
            if normalized is not None:
                return normalized
        return None
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None
        value = decoded if isinstance(decoded, list) else raw.strip("[]").split(",")
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    result: list[str] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            item = " ".join(str(part) for part in item)
        normalized = _string_value(item)
        if normalized:
            result.append(normalized.strip("'\""))
    return list(dict.fromkeys(item for item in result if item))


def _parse_year(value: Any) -> int | None:
    if value is None:
        return None
    import re

    match = re.search(r"(?:19|20)\d{2}", str(value))
    if match is None:
        return None
    year = int(match.group(0))
    return year if 1900 <= year <= 2100 else None


def _clean_doi(value: Any) -> str | None:
    normalized = _string_value(value)
    if not normalized:
        return None
    import re

    normalized = re.sub(r"^https?://doi\.org/", "", normalized, flags=re.I)
    normalized = re.sub(r"^doi:\s*", "", normalized, flags=re.I)
    return normalized.strip() or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PaSa paper_database files to ScholarNavigator local BM25 JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/local_bm25/pasa_papers.jsonl"),
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--zip-member", default=None)
    parser.add_argument("--identity", choices=IDENTITIES, default="arxiv_id")
    parser.add_argument("--max-records", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = convert_corpus(
            args.input,
            args.output,
            identity=args.identity,
            zip_member=args.zip_member,
            report_path=args.report,
            max_records=args.max_records,
        )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
