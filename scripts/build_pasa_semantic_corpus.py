#!/usr/bin/env python3
"""Build a PaSa semantic corpus by exact normalized arXiv ID joins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_ARXIV_ID_RE = re.compile(
    r"^(?:[a-z][a-z0-9.-]+/\d{7}|\d{4,5}\.\d{4,5})$"
)
_ID_FIELDS = (
    "id",
    "arxiv_id",
    "arxivId",
    "arxiv",
    "external_ids.ArXiv",
    "externalids.ArXiv",
)
_TITLE_FIELDS = ("title", "titles", "paper_title", "name")
_ABSTRACT_FIELDS = ("abstract", "summary", "summaries", "paper_abstract")
_CATEGORY_FIELDS = ("categories", "category", "terms", "subjects")
_AUTHOR_FIELDS = ("authors", "author", "authors_parsed")


@dataclass
class BuildReport:
    metadata_path: str
    pasa_paper_index: str
    output_path: str
    metadata_sha256: str
    pasa_paper_index_sha256: str
    input_rows: int
    metadata_valid_id_rows: int
    metadata_missing_id_rows: int
    metadata_missing_title_rows: int
    metadata_missing_abstract_rows: int
    metadata_missing_categories_rows: int
    metadata_missing_authors_rows: int
    duplicate_metadata_rows: int
    conflicting_metadata_ids: int
    pasa_rows: int
    exact_matches: int
    unmatched_metadata_ids: int
    pasa_without_metadata: int
    output_rows: int
    output_abstract_rows: int
    output_category_rows: int
    output_author_rows: int
    field_completeness: dict[str, float]
    coverage: float
    output_sha256: str
    category_counts: dict[str, int]


def build_semantic_corpus(
    metadata_path: Path,
    pasa_paper_index: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
) -> BuildReport:
    _reject_evaluator_path(metadata_path)
    _reject_evaluator_path(pasa_paper_index)
    pasa = _read_pasa_index(pasa_paper_index)
    (
        metadata_rows,
        input_rows,
        missing_id_rows,
        missing_title_rows,
        missing_abstract_rows,
        missing_categories_rows,
        missing_authors_rows,
    ) = (
        _read_metadata(metadata_path)
    )

    metadata_by_id: dict[str, dict[str, Any]] = {}
    duplicate_metadata_rows = 0
    conflicting_ids: set[str] = set()
    for row in metadata_rows:
        arxiv_id = row["arxiv_id"]
        previous = metadata_by_id.get(arxiv_id)
        if previous is None:
            metadata_by_id[arxiv_id] = row
        elif previous == row:
            duplicate_metadata_rows += 1
        else:
            conflicting_ids.add(arxiv_id)
    if conflicting_ids:
        raise ValueError(
            "metadata contains conflicting duplicate arXiv IDs: "
            + ", ".join(sorted(conflicting_ids)[:5])
        )

    matched_ids = sorted(set(pasa) & set(metadata_by_id))
    rows: dict[str, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    for arxiv_id in matched_ids:
        metadata = metadata_by_id[arxiv_id]
        abstract = metadata["abstract"]
        if not abstract:
            continue
        row: dict[str, Any] = {
            "_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": pasa[arxiv_id],
            "abstract": abstract,
        }
        if metadata["categories"]:
            row["categories"] = metadata["categories"]
            category_counts.update(metadata["categories"])
        if metadata["authors"]:
            row["authors"] = metadata["authors"]
        rows[arxiv_id] = row

    if not rows:
        raise ValueError("exact arXiv ID join produced no title+abstract rows")

    output_text = "".join(
        json.dumps(rows[arxiv_id], ensure_ascii=False, sort_keys=True) + "\n"
        for arxiv_id in sorted(rows)
    )
    _atomic_write_text(output_path, output_text)
    report = BuildReport(
        metadata_path=str(metadata_path.resolve()),
        pasa_paper_index=str(pasa_paper_index.resolve()),
        output_path=str(output_path.resolve()),
        metadata_sha256=_sha256_file(metadata_path),
        pasa_paper_index_sha256=_sha256_file(pasa_paper_index),
        input_rows=input_rows,
        metadata_valid_id_rows=len(metadata_rows),
        metadata_missing_id_rows=missing_id_rows,
        metadata_missing_title_rows=missing_title_rows,
        metadata_missing_abstract_rows=missing_abstract_rows,
        metadata_missing_categories_rows=missing_categories_rows,
        metadata_missing_authors_rows=missing_authors_rows,
        duplicate_metadata_rows=duplicate_metadata_rows,
        conflicting_metadata_ids=len(conflicting_ids),
        pasa_rows=len(pasa),
        exact_matches=len(matched_ids),
        unmatched_metadata_ids=len(set(metadata_by_id) - set(pasa)),
        pasa_without_metadata=len(set(pasa) - set(metadata_by_id)),
        output_rows=len(rows),
        output_abstract_rows=sum(bool(row["abstract"]) for row in rows.values()),
        output_category_rows=sum(bool(row.get("categories")) for row in rows.values()),
        output_author_rows=sum(bool(row.get("authors")) for row in rows.values()),
        field_completeness={
            "title": 1.0 if rows else 0.0,
            "abstract": (
                sum(bool(row["abstract"]) for row in rows.values()) / len(rows)
                if rows
                else 0.0
            ),
            "categories": (
                sum(bool(row.get("categories")) for row in rows.values()) / len(rows)
                if rows
                else 0.0
            ),
            "authors": (
                sum(bool(row.get("authors")) for row in rows.values()) / len(rows)
                if rows
                else 0.0
            ),
        },
        coverage=len(rows) / len(pasa) if pasa else 0.0,
        output_sha256=_sha256_bytes(output_text.encode("utf-8")),
        category_counts=dict(category_counts.most_common()),
    )
    if report_path is not None:
        _atomic_write_text(
            report_path,
            json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        )
    return report


def normalize_arxiv_id(value: Any) -> str | None:
    """Return a canonical arXiv ID, removing transport syntax and versions."""

    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.I)
    text = text.removesuffix(".pdf").strip().casefold()
    text = re.sub(r"v\d+$", "", text)
    if not _ARXIV_ID_RE.fullmatch(text):
        return None
    return text


def _read_pasa_index(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("id2paper"), Mapping):
        payload = payload["id2paper"]
    if not isinstance(payload, Mapping):
        raise ValueError("PaSa paper index must be an ID-to-title JSON object")
    result: dict[str, str] = {}
    for raw_id, raw_value in payload.items():
        arxiv_id = normalize_arxiv_id(raw_id)
        title = _clean_text(
            raw_value.get("title") if isinstance(raw_value, Mapping) else raw_value
        )
        if arxiv_id is None or not title:
            raise ValueError("PaSa paper index contains an invalid ID or title")
        if arxiv_id in result:
            raise ValueError(f"PaSa paper index contains duplicate arXiv ID: {arxiv_id}")
        result[arxiv_id] = title
    if not result:
        raise ValueError("PaSa paper index is empty")
    return result


def _read_metadata(
    path: Path,
) -> tuple[list[dict[str, Any]], int, int, int, int, int, int]:
    rows: list[dict[str, Any]] = []
    input_rows = 0
    missing_id_rows = 0
    missing_title_rows = 0
    missing_abstract_rows = 0
    missing_categories_rows = 0
    missing_authors_rows = 0
    for payload in _iter_source_records(path):
        input_rows += 1
        arxiv_id = normalize_arxiv_id(_first_field(payload, _ID_FIELDS))
        if arxiv_id is None:
            missing_id_rows += 1
            continue
        abstract = _clean_text(_first_field(payload, _ABSTRACT_FIELDS))
        title = _clean_text(_first_field(payload, _TITLE_FIELDS))
        categories = _parse_list(_first_field(payload, _CATEGORY_FIELDS))
        authors = _parse_list(_first_field(payload, _AUTHOR_FIELDS))
        missing_title_rows += not bool(title)
        if not abstract:
            missing_abstract_rows += 1
        missing_categories_rows += not bool(categories)
        missing_authors_rows += not bool(authors)
        if not abstract:
            continue
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "authors": authors,
            }
        )
    if not rows:
        raise ValueError(
            "metadata source contains no rows with a valid arXiv ID and abstract"
        )
    return (
        rows,
        input_rows,
        missing_id_rows,
        missing_title_rows,
        missing_abstract_rows,
        missing_categories_rows,
        missing_authors_rows,
    )


def _iter_source_records(path: Path) -> Iterator[Mapping[str, Any]]:
    _reject_evaluator_path(path)
    if path.suffix.casefold() == ".zip":
        if not zipfile.is_zipfile(path):
            raise ValueError("metadata .zip path is not a valid ZIP archive")
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if not name.endswith("/")
                and Path(name).suffix.casefold() in {".csv", ".json", ".jsonl"}
            )
            if len(members) != 1:
                raise ValueError(
                    "metadata archive must contain exactly one CSV, JSON, or JSONL member"
                )
            with archive.open(members[0]) as handle:
                yield from _iter_text_records(handle.read().decode("utf-8"), members[0])
        return
    yield from _iter_text_records(path.read_text(encoding="utf-8"), path.name)


def _iter_text_records(text: str, name: str) -> Iterator[Mapping[str, Any]]:
    suffix = Path(name).suffix.casefold()
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        yield from (row for row in reader if row)
        return
    if suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("metadata JSONL rows must be objects")
                yield payload
        return
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        for key in ("papers", "records", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if all(isinstance(value, Mapping) for value in payload.values()):
                yield from payload.values()
                return
    if isinstance(payload, list):
        if not all(isinstance(value, Mapping) for value in payload):
            raise ValueError("metadata JSON list rows must be objects")
        yield from payload
        return
    raise ValueError("metadata source must contain object records")


def _first_field(payload: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = _lookup_path(payload, name)
        if value not in (None, "", [], {}):
            return value
    return None


def _lookup_path(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        exact = next(
            (key for key in value if str(key).casefold() == part.casefold()),
            None,
        )
        if exact is None:
            return None
        value = value[exact]
    return value


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
        cleaned = _clean_text(item).strip("'\"")
        if cleaned:
            result.append(cleaned)
    return list(dict.fromkeys(result))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\\n", " ").replace("\r", " ").split())


def _reject_evaluator_path(path: Path) -> None:
    normalized = "/".join(path.resolve().parts).casefold()
    if any(
        token in normalized
        for token in ("autoscholarquery", "gold", "qrels")
    ):
        raise ValueError("evaluator gold/qrels paths are not valid corpus inputs")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a PaSa semantic corpus using exact arXiv ID joins."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Cornell/arXiv metadata CSV, JSON, JSONL, or single-member ZIP",
    )
    parser.add_argument(
        "--pasa-paper-index",
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
            args.metadata,
            args.pasa_paper_index,
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
