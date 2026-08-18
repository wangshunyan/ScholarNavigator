from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.evidence_registry import (
    REPOSITORY_ROOT,
    build_evidence_registry,
    canonical_default_contract,
    canonical_default_strategy_ids,
    check_evidence_registry,
    implemented_strategy_ids,
    validate_registry_document,
    write_evidence_registry,
)


MANIFEST_PATH = REPOSITORY_ROOT / "benchmark" / "evidence_registry_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry_bundle() -> tuple[dict, dict, str]:
    return build_evidence_registry(_manifest())


def _kinds(registry: dict) -> set[str]:
    return {
        item["kind"]
        for item in validate_registry_document(registry, repository_root=REPOSITORY_ROOT)
    }


def test_inventory_is_complete_and_has_one_record_per_strategy(registry_bundle) -> None:
    registry, matrix, _summary = registry_bundle
    expected = set(implemented_strategy_ids())
    observed = [item["strategy_id"] for item in registry["strategies"]]

    assert len(expected) == 24
    assert set(observed) == expected
    assert len(observed) == len(set(observed))
    assert matrix["strategy_count"] == 24
    assert matrix["default_enabled_strategy_ids"] == ["current_rules"]
    assert canonical_default_strategy_ids() == ("current_rules",)
    assert canonical_default_contract() == registry["canonical_default_contract"]
    assert registry["canonical_default_contract"]["enable_refchain"] is False
    assert registry["canonical_default_contract"]["default_sources"] == [
        "openalex",
        "arxiv",
        "semantic_scholar",
        "pubmed",
    ]
    assert registry["execution"] == {
        "benchmark_run_count": 0,
        "input_mode": "tracked_repository_evidence_only",
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }


def test_gate_detects_missing_strategy(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    registry["strategies"].pop()
    assert "missing_strategy" in _kinds(registry)


def test_gate_detects_duplicate_and_conflicting_records(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    duplicate = copy.deepcopy(registry["strategies"][0])
    duplicate["conclusion"] = "conflicting conclusion"
    registry["strategies"].append(duplicate)
    kinds = _kinds(registry)
    assert "duplicate_strategy" in kinds
    assert "conflicting_strategy_records" in kinds


def test_gate_detects_evidence_hash_drift(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    registry["strategies"][0]["evidence_sources"][0]["sha256"] = "0" * 64
    assert "evidence_hash_drift" in _kinds(registry)


def test_gate_detects_metric_version_and_artifact_hash_drift(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    registry["strategies"][0]["metric_version"] = "unknown_v9"
    registry["strategies"][4]["artifact_hashes"]["bad"] = "not-a-hash"
    kinds = _kinds(registry)
    assert "metric_version_drift" in kinds
    assert "invalid_artifact_hash" in kinds


def test_blocked_and_unavailable_negative_evidence_remain_valid(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    refchain = next(
        item for item in registry["strategies"] if item["strategy_id"] == "refchain"
    )
    assert refchain["decision"] == "blocked"
    assert refchain["evidence_status"] == "evidence_unavailable"
    assert "tracked_primary_artifact_unavailable" in refchain["blockers"]
    assert validate_registry_document(registry, repository_root=REPOSITORY_ROOT) == []


def test_gate_rejects_default_switch_and_default_without_evidence(registry_bundle) -> None:
    registry = copy.deepcopy(registry_bundle[0])
    experiment = next(
        item
        for item in registry["strategies"]
        if item["strategy_id"] == "lexical_normalization_v1"
    )
    experiment["default_enabled"] = True
    current = next(
        item
        for item in registry["strategies"]
        if item["strategy_id"] == "current_rules"
    )
    current["decision"] = "inconclusive"
    kinds = _kinds(registry)
    assert "default_switch_drift" in kinds
    assert "default_without_passing_evidence" in kinds


def test_two_builds_and_writes_are_byte_identical(tmp_path: Path) -> None:
    first = build_evidence_registry(_manifest())
    second = build_evidence_registry(_manifest())
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_evidence_registry(first_dir, *first)
    write_evidence_registry(second_dir, *second)

    for name in ("registry.json", "matrix.json", "summary.md"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


@pytest.mark.evidence_registry_regression
def test_frozen_evidence_registry_gate(tmp_path: Path) -> None:
    manifest = _manifest()
    if "baseline" not in manifest:
        pytest.skip("baseline is generated after the protocol tests pass")
    report = check_evidence_registry(MANIFEST_PATH, tmp_path / "gate")
    assert report["passed"] is True, report["drifts"][:10]
    assert report["execution"] == {
        "benchmark_run_count": 0,
        "llm_request_count": 0,
        "network_request_count": 0,
        "snapshot_write_count": 0,
    }
