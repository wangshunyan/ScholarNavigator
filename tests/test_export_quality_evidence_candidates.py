from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_quality_evidence_candidates import (
    export_arxiv_candidate_identifiers,
    main,
    write_export,
)


def _run(tmp_path: Path, *, status: str = "succeeded", stage: str = "initial_reranked") -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"run_id": "quality-source"}), encoding="utf-8")
    row = {
        "case_id": "query-1",
        "query": "this text must not be included in the export",
        "status": status,
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": stage,
                    "status": "completed",
                    "candidates": [
                        {"title": "Candidate A", "identifiers": {"arxiv_id": "2401.00001v2"}},
                        {"title": "Candidate B", "identifiers": {"arxiv_id": "2401.00002"}},
                        {"title": "Candidate C", "identifiers": {"arxiv_id": "invalid"}},
                        {"title": "Candidate D", "identifiers": {}},
                    ],
                }
            ],
        },
    }
    (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_exports_only_normalized_arxiv_ids_and_provenance_hashes(tmp_path: Path) -> None:
    run = _run(tmp_path)

    identifiers, report = export_arxiv_candidate_identifiers(run)

    assert identifiers == ["arxiv:2401.00001", "arxiv:2401.00002"]
    assert report["run_id"] == "quality-source"
    assert report["successful_case_count"] == 1
    assert report["stage_candidate_count"] == 4
    assert report["valid_arxiv_identifier_count"] == 2
    assert report["invalid_arxiv_id_count"] == 1
    assert report["skipped_without_arxiv_id_count"] == 1
    assert report["gold_or_query_content_loaded"] is False
    assert len(report["results_sha256"]) == 64


def test_export_rejects_incomplete_or_failed_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="benchmark_results_must_be_successful"):
        export_arxiv_candidate_identifiers(_run(tmp_path / "failed", status="failed"))
    with pytest.raises(ValueError, match="completed_initial_reranked_snapshot_required"):
        export_arxiv_candidate_identifiers(_run(tmp_path / "wrong-stage", stage="initial_judged"))


def test_writer_and_cli_do_not_overwrite_existing_outputs(tmp_path: Path) -> None:
    run = _run(tmp_path)
    identifiers_path = tmp_path / "identifiers.txt"
    report_path = tmp_path / "report.json"
    assert main(
        [
            "--run", str(run),
            "--identifiers-output", str(identifiers_path),
            "--report-output", str(report_path),
        ]
    ) == 0
    assert identifiers_path.read_text(encoding="utf-8") == "arxiv:2401.00001\narxiv:2401.00002\n"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert "this text must not be included" not in json.dumps(written)
    with pytest.raises(ValueError, match="quality_evidence_export_output_exists"):
        write_export(["arxiv:2401.00001"], {}, identifiers_output=identifiers_path, report_output=report_path)
