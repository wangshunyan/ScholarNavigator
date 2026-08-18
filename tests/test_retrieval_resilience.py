from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_retrieval_resilience as resilience_cli  # noqa: E402
from scholar_agent.agents.retriever import (  # noqa: E402
    sanitize_connector_error_text,
)
from scholar_agent.evaluation.retrieval_resilience import (  # noqa: E402
    EXIT_INVARIANT_VIOLATION,
    EXIT_NOT_ELIGIBLE,
    ResilienceNotEligible,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_retrieval_resilience,
    write_json,
)


PROTOCOL = ROOT / "benchmark" / "retrieval_resilience_v1_protocol.json"


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _subset(protocol: dict[str, Any], *names: str) -> dict[str, Any]:
    value = copy.deepcopy(protocol)
    selected = {
        item["scenario"]: item
        for item in value["scenarios"]
        if item["scenario"] in names
    }
    assert set(selected) == set(names)
    value["scenarios"] = [selected[name] for name in names]
    return value


def _scenario(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["scenarios"] if item["scenario"] == name)


@pytest.mark.retrieval_resilience_regression
def test_repository_retrieval_resilience_matrix_passes_offline(
    tmp_path: Path,
) -> None:
    report = run_retrieval_resilience(
        _protocol(), repository_root=ROOT, snapshot_root=tmp_path / "snapshots"
    )
    assert report["status"] == "passed"
    assert report["scenario_count"] == 16
    assert report["violation_count"] == 0
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
        "controlled_fault": None,
    }

    malformed = _scenario(report, "malformed_json_isolated")
    assert malformed["terminal_status"] == "partial_failure"
    assert malformed["candidate_identity_count"] == 4
    assert malformed["candidate_identity_count"] == malformed[
        "expected_preserved_identity_count"
    ]

    partial = _scenario(report, "source_partial_then_failure")
    openalex = next(
        item for item in partial["source_terminals"] if item["source"] == "openalex"
    )
    assert openalex["status"] == "partial_completion"
    assert openalex["returned_record_count"] > 0
    assert partial["candidate_identity_count"] == partial[
        "expected_preserved_identity_count"
    ]

    skipped = _scenario(report, "missing_snapshot_is_not_started")
    semantic_scholar = next(
        item
        for item in skipped["source_terminals"]
        if item["source"] == "semantic_scholar"
    )
    assert semantic_scholar["status"] == "not_started"
    assert semantic_scholar["logical_call_count"] == 0
    assert semantic_scholar["skipped_call_count"] > 0
    assert semantic_scholar["reasons"] == ["snapshot_key_not_recorded"]

    all_failed = _scenario(report, "all_sources_failed")
    assert all_failed["terminal_status"] == "all_sources_failed"
    assert all_failed["candidate_identity_count"] == 0
    assert {item["status"] for item in all_failed["source_terminals"]} == {
        "failed"
    }

    duplicate = _scenario(report, "duplicate_records_deduplicated")
    conflict = _scenario(report, "identity_conflict_kept_separate")
    assert duplicate["candidate_identity_count"] == 5
    assert conflict["candidate_identity_count"] == 7

    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        "sensitive-bearer-value",
        "private-value",
        ".env",
        "/Users/",
        "raw.json",
    ):
        assert forbidden not in serialized


def test_budget_fault_is_detected_with_stable_location(tmp_path: Path) -> None:
    protocol = _subset(_protocol(), "single_source_timeout")
    report = run_retrieval_resilience(
        protocol,
        repository_root=ROOT,
        controlled_fault="budget_overrun",
        snapshot_root=tmp_path,
    )
    assert report["exit_code"] == EXIT_INVARIANT_VIOLATION
    assert report["status"] == "invariant_violation"
    assert report["violation_count"] == 1
    assert report["violations"][0]["invariant"] == (
        "bounded_existing_retry_semantics"
    )
    assert report["violations"][0]["first_difference_path"].endswith(
        ".request_count"
    )


def test_gate_output_is_byte_deterministic(tmp_path: Path) -> None:
    protocol = _subset(
        _protocol(),
        "single_source_rate_limit",
        "pagination_loop_bounded",
        "missing_snapshot_is_not_started",
        "source_partial_then_failure",
        "all_sources_failed",
    )
    first = run_retrieval_resilience(
        protocol, repository_root=ROOT, snapshot_root=tmp_path / "snapshots"
    )
    second = run_retrieval_resilience(
        protocol, repository_root=ROOT, snapshot_root=tmp_path / "snapshots"
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_error_sanitizer_removes_credentials_paths_and_environment_file() -> None:
    value = (
        "unexpected Authorization: Bearer sensitive-bearer-value "
        "api_key=private-value .env /Users/example/private/raw.json"
    )
    sanitized = sanitize_connector_error_text(value)
    for forbidden in (
        "sensitive-bearer-value",
        "private-value",
        ".env",
        "/Users/",
        "raw.json",
    ):
        assert forbidden not in sanitized
    assert "[redacted]" in sanitized
    assert "[environment-file]" in sanitized
    assert "[absolute-path]" in sanitized
    assert sanitize_connector_error_text("http_status:429") == "http_status:429"


def test_fixture_drift_is_not_eligible(tmp_path: Path) -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["fixture"]["path"] = "datasets/eval_fixtures/resilience/missing.json"
    temp_protocol = tmp_path / "protocol.json"
    write_json(temp_protocol, protocol)
    with pytest.raises(ResilienceNotEligible, match="fixture_missing"):
        load_protocol(temp_protocol, repository_root=tmp_path)


def test_frozen_baselines_are_structurally_not_eligible() -> None:
    report = audit_frozen_baseline_eligibility(
        _protocol(), repository_root=ROOT
    )
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["status"] == "not_eligible"
    assert report["profile_count"] > 0
    assert {item["status"] for item in report["profiles"]} == {"not_eligible"}
    assert {item["reason"] for item in report["profiles"]} == {
        "source_level_fault_replay_metadata_unavailable"
    }


def test_cli_uses_contract_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resilience_cli.main([]) == 4

    monkeypatch.setattr(
        resilience_cli,
        "run_retrieval_resilience",
        lambda *_args, **_kwargs: {
            "schema_version": "1",
            "contract": "retrieval_resilience_v1",
            "gate": "retrieval_resilience_gate",
            "status": "invariant_violation",
            "exit_code": 2,
        },
    )
    assert (
        resilience_cli.main(
            [
                "--repository-root",
                str(ROOT),
                "--protocol",
                str(PROTOCOL),
                "check",
            ]
        )
        == 2
    )
    assert (
        resilience_cli.main(
            [
                "--repository-root",
                str(ROOT),
                "--protocol",
                str(PROTOCOL),
                "audit-frozen",
            ]
        )
        == 3
    )
