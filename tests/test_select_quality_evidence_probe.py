from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.select_quality_evidence_probe import (
    select_probe_identifiers,
    write_probe_selection,
)


def _candidates(tmp_path: Path, values: list[str]) -> Path:
    path = tmp_path / "candidate-arxiv-ids.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return path


def test_selects_deterministic_hash_ranked_probe_without_query_content(
    tmp_path: Path,
) -> None:
    candidates = _candidates(
        tmp_path,
        [f"arxiv:2401.{number:05d}" for number in range(1, 25)],
    )

    first, first_report = select_probe_identifiers(candidates, limit=20)
    second, second_report = select_probe_identifiers(candidates, limit=20)

    assert first == second
    assert len(first) == 20
    assert first_report == second_report
    assert first_report["candidate_identifiers_sha256"] == hashlib.sha256(
        candidates.read_bytes()
    ).hexdigest()
    assert first_report["gold_or_query_content_loaded"] is False
    assert "this text must not be included" not in json.dumps(first_report)


def test_probe_selector_rejects_invalid_population_or_limit(tmp_path: Path) -> None:
    too_small = _candidates(tmp_path / "small", ["arxiv:2401.00001"])
    with pytest.raises(ValueError, match="quality_evidence_probe_population_too_small"):
        select_probe_identifiers(too_small, limit=2)

    enough = _candidates(
        tmp_path / "enough",
        [f"arxiv:2401.{number:05d}" for number in range(1, 21)],
    )
    with pytest.raises(ValueError, match="quality_evidence_probe_limit_out_of_range"):
        select_probe_identifiers(enough, limit=21)

    invalid = _candidates(
        tmp_path / "invalid",
        [f"arxiv:2401.{number:05d}" for number in range(1, 20)] + ["doi:10.1/example"],
    )
    with pytest.raises(ValueError, match="quality_evidence_candidate_arxiv_identifier_required"):
        select_probe_identifiers(invalid, limit=20)


def test_probe_writer_records_selected_identity_and_refuses_overwrite(tmp_path: Path) -> None:
    candidates = _candidates(
        tmp_path,
        [f"arxiv:2401.{number:05d}" for number in range(1, 21)],
    )
    identifiers, report = select_probe_identifiers(candidates)
    identifier_output = tmp_path / "probe.txt"
    report_output = tmp_path / "probe-report.json"

    write_probe_selection(
        identifiers,
        report,
        identifiers_output=identifier_output,
        report_output=report_output,
    )

    written = json.loads(report_output.read_text(encoding="utf-8"))
    assert written["selected_identifiers_sha256"] == hashlib.sha256(
        identifier_output.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="quality_evidence_probe_output_exists"):
        write_probe_selection(
            identifiers,
            report,
            identifiers_output=identifier_output,
            report_output=report_output,
        )
