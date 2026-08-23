#!/usr/bin/env python3
"""Merge an external, legally obtained metadata JSONL into a local paper corpus.

The tool is gold-blind and offline.  It never fetches metadata and never
guesses missing publication fields.  The base corpus remains authoritative for
identity and existing non-empty values unless --overwrite is explicitly used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

IDENTITY = "arxiv_id"
FIELDS = ("title", "abstract", "authors", "year", "venue", "doi")
ARXIV_RE = re.compile(r"^(?:[a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.I)


class MetadataMergeError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MetadataMergeError(f"invalid_jsonl:{path.name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise MetadataMergeError(f"row_not_object:{path.name}:{line_number}")
            rows.append(value)
    return rows


def _identity(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized or not ARXIV_RE.fullmatch(normalized):
        raise MetadataMergeError(f"invalid_arxiv_id:{normalized or '<empty>'}")
    return re.sub(r"v\d+$", "", normalized, flags=re.IGNORECASE).casefold()


def _clean_field(field: str, value: Any) -> Any:
    if field in {"title", "abstract", "venue", "doi"}:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise MetadataMergeError(f"field_not_string:{field}")
        return " ".join(value.split())
    if field == "authors":
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value):
            return [item.strip() for item in value]
        raise MetadataMergeError("field_invalid:authors")
    if field == "year":
        if value in (None, ""):
            return None
        if isinstance(value, bool) or not isinstance(value, int) or not 1900 <= value <= 2100:
            raise MetadataMergeError("field_invalid:year")
        return value
    raise MetadataMergeError(f"unsupported_field:{field}")


def _value_present(value: Any) -> bool:
    return value not in (None, "", []) and value != {}


def merge_rows(
    base_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    *,
    overwrite: bool = False,
    reject_conflicts: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        key = _identity(row.get(IDENTITY))
        if key in by_id:
            raise MetadataMergeError(f"duplicate_base_id:{key}")
        normalized = dict(row)
        normalized[IDENTITY] = re.sub(
            r"v\d+$", "", str(row[IDENTITY]).strip(), flags=re.IGNORECASE
        )
        by_id[key] = normalized

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        key = _identity(row.get(IDENTITY))
        if key in metadata_by_id:
            raise MetadataMergeError(f"duplicate_metadata_id:{key}")
        metadata_by_id[key] = row

    conflicts: list[dict[str, str]] = []
    updated = 0
    unmatched = 0
    filled: dict[str, int] = {field: 0 for field in FIELDS}
    for key, metadata in metadata_by_id.items():
        target = by_id.get(key)
        if target is None:
            unmatched += 1
            continue
        changed = False
        for field in FIELDS:
            if field not in metadata:
                continue
            incoming = _clean_field(field, metadata[field])
            existing = _clean_field(field, target.get(field))
            if not _value_present(incoming):
                continue
            if _value_present(existing) and existing != incoming:
                conflicts.append({"arxiv_id": key, "field": field})
                if reject_conflicts:
                    raise MetadataMergeError(f"conflict:{key}:{field}")
                if not overwrite:
                    continue
            if existing != incoming:
                target[field] = incoming
                filled[field] += 1
                changed = True
        if changed:
            updated += 1

    output = [by_id[_identity(row[IDENTITY])] for row in base_rows]
    report = {
        "schema_version": "paper-metadata-merge-v1",
        "base_count": len(base_rows),
        "metadata_count": len(metadata_rows),
        "matched_count": len(metadata_by_id) - unmatched,
        "unmatched_metadata_count": unmatched,
        "updated_document_count": updated,
        "filled_field_counts": filled,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:100],
        "overwrite": overwrite,
        "reject_conflicts": reject_conflicts,
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reject-conflicts", action="store_true")
    args = parser.parse_args()
    try:
        base = _read_jsonl(args.base)
        metadata = _read_jsonl(args.metadata)
        merged, report = merge_rows(
            base, metadata, overwrite=args.overwrite, reject_conflicts=args.reject_conflicts
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    except MetadataMergeError as exc:
        print(json.dumps({"schema_version": "paper-metadata-merge-v1", "status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
