#!/usr/bin/env python3
"""Create a safe, portable evidence bundle from a completed server run.

This tool is intentionally offline.  It never contacts a server and never
reads dotenv or SSH files.  The operator runs it *on the experiment host* and
copies the resulting archive to a local staging directory.  Only the
allow-listed run artifacts are included; absolute paths, credentials and host
identifiers are rejected.  Raw bundles belong in ignored ``outputs/`` storage,
not in Git history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    BenchmarkRunCommitStore,
    CrashConsistencyError,
)


SCHEMA = "server_evidence_bundle_v1"
LEGACY_INVENTORY_SCHEMA = "server_legacy_inventory_v1"
ALLOWED_FILES = (
    "config.json",
    "metrics.json",
    "results.jsonl",
    "resource_ledger.json",
    "summary.md",
    "stage_metrics.json",
    "error_analysis.json",
    ".run_complete",
    ".run_committed",
)
LEGACY_INVENTORY_FILES = (
    "config.json",
    "metrics.json",
    "resource_ledger.json",
    "summary.md",
    "stage_metrics.json",
    "error_analysis.json",
    "dataset_report.json",
    "qualification_gate.json",
    "reranker_audit.json",
    "failures.jsonl",
)
REQUIRED_FILES = {"config.json", "metrics.json", "results.jsonl", "resource_ledger.json"}
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
FORBIDDEN_TEXT = (
    re.compile(r"(?i)(?:^|[^0-9])172\.16\.36\.16(?:[^0-9]|$)"),
    re.compile(r"(?i)(?:^|[^a-z])(?:ssh[_ -]?private|private[_ -]?key|api[_ -]?key|access[_ -]?token|secret)(?:[^a-z]|$)"),
    re.compile(r"(?i)(?:^|[\\/])(?:users|home|mnt|root)[\\/][^\s\"']+"),
)


class EvidenceError(ValueError):
    """A bundle is not eligible for portable evidence handoff."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_TEXT)


def _safe_text(path: Path) -> None:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise EvidenceError(f"file_too_large:{path.name}")
    # Evidence files are JSON/JSONL/Markdown.  Decode only for the safety
    # scan; the original bytes and hash are retained in the archive.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"non_utf8_evidence:{path.name}") from exc
    # Content is redacted before packaging.  This early check only guarantees
    # that the file is a bounded, UTF-8 text artifact we can safely transform.


def _redact_string(value: str) -> str:
    value = re.sub(r"(?i)(?:^|[^0-9])172\.16\.36\.16(?:[^0-9]|$)", "<redacted-host>", value)
    value = re.sub(r"(?:(?:[A-Za-z]:)?[\\/](?:Users|home|mnt|root)[\\/][^\s\"']+)", "<redacted-path>", value)
    # Run results can legitimately retain diagnostic messages such as
    # "API key unavailable" or a copied private-key error.  They are useful
    # to the local operator but must never leave the experiment host.  Replace
    # the entire string rather than attempting token-level cleanup, so that a
    # real credential cannot survive next to the matched phrase.
    if _contains_sensitive_text(value):
        return "<redacted-sensitive-text>"
    return value


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {"api_key", "token", "password", "secret", "private_key"}:
                raise EvidenceError(f"credential_field_present:{key}")
            result[key] = _redact_value(child)
        return result
    if isinstance(value, list):
        return [_redact_value(child) for child in value]
    return _redact_string(value) if isinstance(value, str) else value


def _redacted_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".md" or path.name.startswith(".run_"):
        result = _redact_string(raw.decode("utf-8"))
        if _contains_sensitive_text(result):
            raise EvidenceError(f"sensitive_text_in_evidence:{path.name}")
        return result.encode("utf-8")
    if path.name == "results.jsonl":
        rows = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"invalid_jsonl:{path.name}:{line_number}") from exc
            if not isinstance(value, dict):
                raise EvidenceError(f"jsonl_row_not_object:{path.name}:{line_number}")
            rows.append(json.dumps(_redact_value(value), ensure_ascii=False, sort_keys=True))
        result = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    else:
        try:
            value = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"invalid_json:{path.name}") from exc
        if not isinstance(value, (dict, list)):
            raise EvidenceError(f"json_root_invalid:{path.name}")
        result = (json.dumps(_redact_value(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if _contains_sensitive_text(result.decode("utf-8")):
        raise EvidenceError(f"sensitive_text_in_evidence:{path.name}")
    return result


def _validate_json_shape(path: Path) -> None:
    if path.suffix == ".md" or path.name.startswith(".run_"):
        return
    if path.name == "results.jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceError(f"invalid_jsonl:{path.name}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise EvidenceError(f"jsonl_row_not_object:{path.name}:{line_number}")
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid_json:{path.name}") from exc
    if not isinstance(value, (dict, list)):
        raise EvidenceError(f"json_root_invalid:{path.name}")


def _committed_generation_completion(run_dir: Path) -> dict[str, Any] | None:
    """Verify the authoritative completion chain used by new benchmark runs.

    ``run_benchmark.py`` stores the completion marker below ``.run_commits``
    rather than creating a legacy top-level marker.  Do not merely discover
    that nested file: load the committed generation and ensure the public
    compatibility view still exactly equals its authoritative artifacts.
    """

    store = BenchmarkRunCommitStore(run_dir)
    if not store.root.exists():
        return None
    try:
        state = store.load_latest()
    except (CrashConsistencyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("committed_generation_invalid") from exc
    if state.status != "completed":
        return None
    marker = state.generation_path / "RUN_COMPLETED"
    if not marker.is_file():
        return None
    expected = store.public_artifacts(state)
    for name in REQUIRED_FILES:
        path = run_dir / name
        if not path.is_file() or expected.get(name) != path.read_bytes():
            raise EvidenceError(f"committed_compatibility_view_drift:{name}")
    return {
        "kind": "run_commit_generation",
        "generation": state.generation,
        "record_count": len(state.records),
        "completion_marker_sha256": _sha256(marker),
    }


def inspect_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise EvidenceError(f"run_directory_missing:{run_dir.name}")
    present: list[dict[str, Any]] = []
    for name in ALLOWED_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        if not path.is_file() or PurePosixPath(name).name != name:
            raise EvidenceError(f"invalid_member:{name}")
        _safe_text(path)
        _validate_json_shape(path)
        exported = _redacted_bytes(path)
        present.append({
            "path": name,
            "source_bytes": path.stat().st_size,
            "source_sha256": _sha256(path),
            "exported_bytes": len(exported),
            "exported_sha256": hashlib.sha256(exported).hexdigest(),
        })
    missing = sorted(REQUIRED_FILES - {row["path"] for row in present})
    if missing:
        raise EvidenceError("required_artifact_missing:" + ",".join(missing))
    markers = sorted(
        {row["path"] for row in present} & {".run_complete", ".run_committed"}
    )
    committed_completion = _committed_generation_completion(run_dir)
    if committed_completion is not None:
        markers.append("run_commit_generation")
    if not markers:
        raise EvidenceError("completion_marker_missing")
    result: dict[str, Any] = {
        "run_id": run_dir.name,
        "files": present,
        "required_files": sorted(REQUIRED_FILES),
        "completion_markers": sorted(markers),
        "source_path_recorded": False,
        "sensitive_values_scanned": True,
    }
    if committed_completion is not None:
        result["committed_completion"] = committed_completion
    return result


def inspect_legacy_inventory(run_dir: Path) -> dict[str, Any]:
    """Inspect an old run without asserting it ever reached completion.

    Legacy directories sometimes predate completion markers.  This inventory
    deliberately excludes raw ``results.jsonl`` and any gold diagnostics; it
    records only the raw result artifact's size and digest so the local audit
    can later request the exact payload if the run is otherwise usable.
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise EvidenceError(f"run_directory_missing:{run_dir.name}")
    present: list[dict[str, Any]] = []
    for name in LEGACY_INVENTORY_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        if not path.is_file() or PurePosixPath(name).name != name:
            raise EvidenceError(f"invalid_member:{name}")
        _safe_text(path)
        _validate_json_shape(path)
        exported = _redacted_bytes(path)
        present.append({
            "path": name,
            "source_bytes": path.stat().st_size,
            "source_sha256": _sha256(path),
            "exported_bytes": len(exported),
            "exported_sha256": hashlib.sha256(exported).hexdigest(),
        })
    required_inventory = REQUIRED_FILES - {"results.jsonl"}
    missing = sorted(required_inventory - {row["path"] for row in present})
    if missing:
        raise EvidenceError("required_artifact_missing:" + ",".join(missing))
    results = run_dir / "results.jsonl"
    if not results.is_file():
        raise EvidenceError("required_artifact_missing:results.jsonl")
    if results.stat().st_size > MAX_FILE_BYTES:
        raise EvidenceError("file_too_large:results.jsonl")
    markers = sorted(
        name for name in (".run_complete", ".run_committed") if (run_dir / name).is_file()
    )
    return {
        "run_id": run_dir.name,
        "files": present,
        "required_files": sorted(required_inventory),
        "result_artifact": {
            "path": "results.jsonl",
            "source_bytes": results.stat().st_size,
            "source_sha256": _sha256(results),
            "exported": False,
        },
        "completion_markers": markers,
        "completion_status": "unverified_legacy_inventory",
        "source_path_recorded": False,
        "sensitive_values_scanned": True,
        "gold_diagnostics_exported": False,
    }


def build_bundle(run_dir: Path, output: Path) -> dict[str, Any]:
    inspection = inspect_run(run_dir)
    root = run_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == root or root in output.parents:
        raise EvidenceError("output_must_be_outside_run")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run": inspection,
        "archive_contains_redacted_bytes": True,
        "official_metric_scope": "internal_engineering_only",
        "network_requests": 0,
        "ssh_or_dotenv_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in inspection["files"]:
            name = str(row["path"])
            archive.writestr(f"run/{name}", _redacted_bytes(root / name))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    return {**manifest, "archive": output.name, "archive_sha256": _sha256(output)}


def build_legacy_inventory(run_dir: Path, output: Path) -> dict[str, Any]:
    inspection = inspect_legacy_inventory(run_dir)
    root = run_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if output == root or root in output.parents:
        raise EvidenceError("output_must_be_outside_run")
    manifest: dict[str, Any] = {
        "schema_version": LEGACY_INVENTORY_SCHEMA,
        "run": inspection,
        "archive_contains_redacted_bytes": True,
        "official_metric_scope": "unverified_legacy_inventory_not_official",
        "raw_results_exported": False,
        "gold_diagnostics_exported": False,
        "network_requests": 0,
        "ssh_or_dotenv_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in inspection["files"]:
            name = str(row["path"])
            archive.writestr(f"run/{name}", _redacted_bytes(root / name))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    return {**manifest, "archive": output.name, "archive_sha256": _sha256(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--legacy-inventory",
        action="store_true",
        help="export a non-official inventory for a pre-marker legacy run; raw results stay on the server",
    )
    args = parser.parse_args()
    try:
        builder = build_legacy_inventory if args.legacy_inventory else build_bundle
        print(json.dumps(builder(args.run_dir, args.output), ensure_ascii=False, sort_keys=True))
    except EvidenceError as exc:
        schema = LEGACY_INVENTORY_SCHEMA if args.legacy_inventory else SCHEMA
        print(json.dumps({"schema_version": schema, "status": "blocked", "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
