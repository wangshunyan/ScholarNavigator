"""Streaming, gold-blind metadata audit for local paper JSONL corpora."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scholar_agent.core.identity import normalize_arxiv_id


REQUIRED_FIELDS = ("title", "abstract", "authors", "year", "venue", "doi")


def audit_jsonl_corpus(
    path: str | Path,
    *,
    identity_field: str = "arxiv_id",
    required_fields: Sequence[str] = (),
) -> dict[str, Any]:
    unknown_fields = sorted(set(required_fields) - set(REQUIRED_FIELDS))
    if unknown_fields:
        raise ValueError(f"unsupported_required_metadata_fields:{','.join(unknown_fields)}")
    required = tuple(dict.fromkeys(required_fields))
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    row_count = 0
    invalid_json_rows = 0
    non_object_rows = 0
    duplicate_identity_rows = 0
    invalid_identity_rows = 0
    identities: set[str] = set()
    present_counts = {field: 0 for field in REQUIRED_FIELDS}
    with source.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            row_count += 1
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid_json_rows += 1
                continue
            if not isinstance(value, dict):
                non_object_rows += 1
                continue
            for field in REQUIRED_FIELDS:
                item = value.get(field)
                if item not in (None, "", [], {}):
                    present_counts[field] += 1
            raw_identity = value.get(identity_field)
            identity = normalize_arxiv_id(str(raw_identity)) if raw_identity else None
            if identity is None:
                invalid_identity_rows += 1
            elif identity in identities:
                duplicate_identity_rows += 1
            else:
                identities.add(identity)
    valid_rows = row_count - invalid_json_rows - non_object_rows
    required_fields_complete = all(
        valid_rows > 0 and present_counts[field] == valid_rows for field in required
    )
    structural_passed = (
        bool(valid_rows)
        and invalid_json_rows == 0
        and non_object_rows == 0
        and invalid_identity_rows == 0
        and duplicate_identity_rows == 0
    )
    return {
        "schema_version": "corpus-metadata-audit-v1",
        "path": str(source),
        "sha256": digest.hexdigest(),
        "identity_field": identity_field,
        "row_count": row_count,
        "valid_object_rows": valid_rows,
        "unique_identity_count": len(identities),
        "invalid_json_rows": invalid_json_rows,
        "non_object_rows": non_object_rows,
        "invalid_identity_rows": invalid_identity_rows,
        "duplicate_identity_rows": duplicate_identity_rows,
        "present_counts": present_counts,
        "field_completeness": {
            field: (present_counts[field] / valid_rows if valid_rows else 0.0)
            for field in REQUIRED_FIELDS
        },
        "required_fields": list(required),
        "required_fields_complete": required_fields_complete,
        "structural_passed": structural_passed,
        "passed": structural_passed and required_fields_complete,
    }
