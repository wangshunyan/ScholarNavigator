#!/usr/bin/env python3
"""Export exact arXiv candidate IDs from a completed benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.core.identity import normalize_arxiv_id  # noqa: E402


_STAGE = "initial_reranked"
_STRICT_ARXIV_ID_RE = re.compile(
    r"^(?:[a-z][a-z0-9.-]+/\d{7}|\d{4,5}\.\d{4,5})$"
)


def export_arxiv_candidate_identifiers(
    run_dir: Path,
    *,
    results_stream: BinaryIO | None = None,
    expected_results_sha256: str | None = None,
    source_run_id: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Read only completed candidate snapshots; never load evaluation gold data."""

    root = run_dir.expanduser().resolve()
    config_path = root / "config.json"
    results_path = root / "results.jsonl"
    if not config_path.is_file() or (results_stream is None and not results_path.is_file()):
        raise ValueError("benchmark_run_artifacts_required")
    config = _read_object(config_path)
    if results_stream is None:
        if source_run_id is not None:
            raise ValueError("source_run_id_requires_stdin")
        rows = _read_jsonl(results_path)
        results_sha256 = _sha256(results_path)
        results_transport = "local_file"
        run_id = str(config.get("run_id") or root.name)
    else:
        if not _is_sha256(expected_results_sha256):
            raise ValueError("streamed_results_sha256_required")
        if not _is_run_id(source_run_id):
            raise ValueError("streamed_source_run_id_required")
        stream_digest = hashlib.sha256()
        rows = _read_jsonl_stream(results_stream, stream_digest)
        results_sha256 = None
        results_transport = "stdin_verified_stream"
        run_id = source_run_id

    identifiers: set[str] = set()
    successful_case_count = 0
    candidate_count = 0
    skipped_without_arxiv_id = 0
    invalid_arxiv_id_count = 0
    for row in rows:
        if row.get("status") != "succeeded":
            raise ValueError("benchmark_results_must_be_successful")
        successful_case_count += 1
        snapshot = _stage_snapshot(row, _STAGE)
        for candidate in _mapping_list(snapshot.get("candidates")):
            candidate_count += 1
            raw = _value(_value(candidate, "identifiers", {}), "arxiv_id")
            if raw in (None, ""):
                skipped_without_arxiv_id += 1
                continue
            normalized = normalize_arxiv_id(str(raw))
            if normalized is None or not _STRICT_ARXIV_ID_RE.fullmatch(normalized):
                invalid_arxiv_id_count += 1
                continue
            identifiers.add(f"arxiv:{normalized}")
    if successful_case_count == 0:
        raise ValueError("benchmark_results_empty")
    if results_stream is not None:
        results_sha256 = stream_digest.hexdigest()
        if results_sha256 != expected_results_sha256:
            raise ValueError("streamed_results_sha256_mismatch")
    if not identifiers:
        raise ValueError("benchmark_candidates_without_valid_arxiv_id")
    report = {
        "schema_version": "quality-evidence-candidate-export-v1",
        "source_kind": "completed_run_initial_reranked_candidates_only",
        "run_id": run_id,
        "config_sha256": _sha256(config_path),
        "results_sha256": results_sha256,
        "results_transport": results_transport,
        "successful_case_count": successful_case_count,
        "stage": _STAGE,
        "stage_candidate_count": candidate_count,
        "valid_arxiv_identifier_count": len(identifiers),
        "skipped_without_arxiv_id_count": skipped_without_arxiv_id,
        "invalid_arxiv_id_count": invalid_arxiv_id_count,
        "gold_or_query_content_loaded": False,
    }
    return sorted(identifiers), report


def write_export(
    identifiers: Sequence[str],
    report: Mapping[str, Any],
    *,
    identifiers_output: Path,
    report_output: Path,
) -> None:
    """Write new outputs only, so historical evidence cannot be overwritten."""

    if identifiers_output.exists() or report_output.exists():
        raise ValueError("quality_evidence_export_output_exists")
    identifiers_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    identifiers_output.write_text("".join(f"{item}\n" for item in identifiers), encoding="utf-8")
    report_with_identifier_identity = {
        **dict(report),
        "identifiers_sha256": _sha256(identifiers_output),
    }
    report_output.write_text(
        json.dumps(report_with_identifier_identity, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _stage_snapshot(row: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    diagnostics = _value(row, "stage_diagnostics", {})
    snapshots = _mapping_list(_value(diagnostics, "snapshots", []))
    matches = [item for item in snapshots if item.get("stage") == stage]
    if len(matches) != 1 or matches[0].get("status") != "completed":
        raise ValueError("completed_initial_reranked_snapshot_required")
    return matches[0]


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark_config_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark_config_object_required")
    return value


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("benchmark_results_unavailable") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"benchmark_results_invalid_line:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"benchmark_results_object_required:{number}")
        yield value


def _read_jsonl_stream(
    stream: BinaryIO,
    digest: Any,
) -> Iterable[Mapping[str, Any]]:
    for number, raw_line in enumerate(stream, start=1):
        digest.update(raw_line)
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"benchmark_results_invalid_utf8_line:{number}") from exc
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"benchmark_results_invalid_line:{number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"benchmark_results_object_required:{number}")
        yield value


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _value(value: object, key: str, default: object = None) -> object:
    return value.get(key, default) if isinstance(value, Mapping) else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str | None) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _is_run_id(value: str | None) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--results-stdin",
        action="store_true",
        help="read raw results.jsonl bytes from stdin without writing them locally",
    )
    parser.add_argument(
        "--expected-results-sha256",
        default=None,
        help="required SHA-256 for --results-stdin",
    )
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="required completed benchmark RunId for --results-stdin",
    )
    parser.add_argument("--identifiers-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.results_stdin:
            identifiers, report = export_arxiv_candidate_identifiers(
                args.run,
                results_stream=sys.stdin.buffer,
                expected_results_sha256=args.expected_results_sha256,
                source_run_id=args.source_run_id,
            )
        elif args.expected_results_sha256 is not None or args.source_run_id is not None:
            raise ValueError("streamed_results_arguments_require_stdin")
        else:
            identifiers, report = export_arxiv_candidate_identifiers(args.run)
        write_export(
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
