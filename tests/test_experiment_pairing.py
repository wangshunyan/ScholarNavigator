from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import run_benchmark
from scholar_agent.core.evaluation_schemas import EvalQuery
from scholar_agent.evaluation.crash_consistency import BenchmarkRunCommitStore
from scholar_agent.evaluation.experiment_pairing import (
    EXIT_NOT_READY,
    EXIT_PASSED,
    EXIT_VIOLATION,
    ComparisonPlanV1,
    TreatmentChange,
    audit_evidence_registry,
    audit_frozen_eligibility,
    build_local_fixture,
    deterministic_fixture_report,
    load_comparison_plan,
    validate_pairing,
)
from scholar_agent.evaluation.run_provenance import RunManifestV1
from scholar_agent.evaluation.snapshot_resume import stable_hash


pytestmark = pytest.mark.experiment_pairing_integrity_regression


def _run(root: Path, **kwargs: object) -> dict[str, object]:
    plan, baseline, candidate = build_local_fixture(root, **kwargs)
    return validate_pairing(
        plan, baseline, candidate, repository_root=root
    )


def _invariants(report: dict[str, object]) -> set[str]:
    return {
        str(item["invariant"])
        for item in report.get("violations", [])  # type: ignore[union-attr]
    }


def _rewrite_manifest(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_legal_single_treatment_pair_passes_and_is_byte_deterministic() -> None:
    first = deterministic_fixture_report()
    second = deterministic_fixture_report()
    assert first == second
    assert first["exit_code"] == EXIT_PASSED
    assert first["paired_query_count"] == 3
    assert first["observation"] == {
        "plan_binding_affects_execution": False,
        "result_payload_compared_for_quality": False,
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
    }


def test_multiple_exact_treatments_are_allowed_but_parent_path_is_rejected(
    tmp_path: Path,
) -> None:
    plan_path, _, _ = build_local_fixture(tmp_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["allowed_treatment_changes"] = [
        {
            "pointer": "/configuration/values/feature_a",
            "baseline_value": False,
            "candidate_value": True,
        },
        {
            "pointer": "/configuration/values/feature_b",
            "baseline_value": "off",
            "candidate_value": "on",
        },
    ]
    values = payload["common_execution_contract"]["configuration"]["values"]
    values.pop("treatment_mode")
    values["feature_a"] = {
        "declared_treatment": "/configuration/values/feature_a"
    }
    values["feature_b"] = {
        "declared_treatment": "/configuration/values/feature_b"
    }
    payload["common_execution_contract_sha256"] = stable_hash(
        payload["common_execution_contract"]
    )
    assert len(ComparisonPlanV1.model_validate(payload).allowed_treatment_changes) == 2
    with pytest.raises(ValidationError):
        TreatmentChange(
            pointer="/configuration/values",
            baseline_value="off",
            candidate_value="on",
        )


def test_hidden_treatment_is_a_stable_violation() -> None:
    report = deterministic_fixture_report(controlled_fault="hidden_treatment")
    assert report["exit_code"] == EXIT_VIOLATION
    assert _invariants(report) == {"undeclared_execution_contract_drift"}
    assert report == deterministic_fixture_report(controlled_fault="hidden_treatment")


def test_declared_treatment_must_actually_be_applied(tmp_path: Path) -> None:
    plan, baseline, candidate = build_local_fixture(tmp_path)

    def remove_treatment(payload: dict[str, object]) -> None:
        configuration = payload["configuration"]  # type: ignore[assignment]
        configuration["values"]["treatment_mode"] = "off"  # type: ignore[index]
        configuration["summary_sha256"] = stable_hash(  # type: ignore[index]
            {
                "sources": configuration["sources"],  # type: ignore[index]
                "budgets": configuration["budgets"],  # type: ignore[index]
                "values": configuration["values"],  # type: ignore[index]
            }
        )

    _rewrite_manifest(candidate, remove_treatment)
    report = validate_pairing(plan, baseline, candidate, repository_root=tmp_path)
    assert report["exit_code"] == EXIT_VIOLATION
    assert "declared_treatment_missing_or_drifted" in _invariants(report)


@pytest.mark.parametrize("field", ["prompt", "budget", "seed"])
def test_prompt_budget_and_seed_drift_fail(tmp_path: Path, field: str) -> None:
    plan, baseline, candidate = build_local_fixture(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        if field == "prompt":
            payload["prompt"]["versions"]["planner"] = "drifted"  # type: ignore[index]
        elif field == "budget":
            configuration = payload["configuration"]  # type: ignore[assignment]
            configuration["budgets"]["top_k"] = 99  # type: ignore[index]
            configuration["summary_sha256"] = stable_hash(
                {
                    "sources": configuration["sources"],  # type: ignore[index]
                    "budgets": configuration["budgets"],  # type: ignore[index]
                    "values": configuration["values"],  # type: ignore[index]
                }
            )
        else:
            determinism = payload["determinism"]  # type: ignore[assignment]
            determinism["random_seed"] = 99  # type: ignore[index]
            determinism["summary_sha256"] = stable_hash(  # type: ignore[index]
                {
                    "random_seed": 99,
                    "parameters": determinism["parameters"],  # type: ignore[index]
                }
            )

    _rewrite_manifest(candidate, mutate)
    report = validate_pairing(plan, baseline, candidate, repository_root=tmp_path)
    assert report["exit_code"] == EXIT_VIOLATION


def test_query_reordering_is_not_masked(tmp_path: Path) -> None:
    report = _run(tmp_path, candidate_query_order=(1, 0, 2))
    assert report["exit_code"] == EXIT_VIOLATION
    assert "candidate_query_population_mismatch" in _invariants(report)


def test_single_side_missing_and_duplicate_records_fail(tmp_path: Path) -> None:
    missing = deterministic_fixture_report(controlled_fault="asymmetric_coverage")
    assert missing["exit_code"] == EXIT_VIOLATION
    assert "asymmetric_query_coverage" in _invariants(missing)

    duplicate_root = tmp_path / "duplicate"
    duplicate = _run(duplicate_root, duplicate_candidate_index=0)
    assert duplicate["exit_code"] == EXIT_VIOLATION
    assert "duplicate_query_record" in _invariants(duplicate)


def test_asymmetric_failure_or_cancellation_fails(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        candidate_statuses=("succeeded", "cancelled", "succeeded"),
    )
    assert report["exit_code"] == EXIT_VIOLATION
    assert "asymmetric_terminal_status" in _invariants(report)


def test_predeclared_exclusion_is_symmetric_and_posthoc_exclusion_fails(
    tmp_path: Path,
) -> None:
    legal = _run(
        tmp_path / "legal",
        baseline_statuses=("succeeded", "excluded", "succeeded"),
        candidate_statuses=("succeeded", "excluded", "succeeded"),
        excluded_indexes=(1,),
    )
    assert legal["exit_code"] == EXIT_PASSED

    posthoc = _run(
        tmp_path / "posthoc",
        baseline_statuses=("succeeded", "excluded", "succeeded"),
        candidate_statuses=("succeeded", "excluded", "succeeded"),
    )
    assert posthoc["exit_code"] == EXIT_VIOLATION
    assert "post_hoc_exclusion" in _invariants(posthoc)


def test_symmetric_intermediate_is_not_ready_and_cannot_claim_common_success(
    tmp_path: Path,
) -> None:
    report = _run(tmp_path, partial_count=2)
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["status"] == "not_ready"
    assert report["paired_query_count"] == 2


def test_plan_hash_and_manifest_role_mismatch_fail(tmp_path: Path) -> None:
    plan, baseline, candidate = build_local_fixture(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["plan_id"] = "tampered"
    plan.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    report = validate_pairing(plan, baseline, candidate, repository_root=tmp_path)
    assert report["exit_code"] == EXIT_VIOLATION
    assert "run_manifest_invalid" in _invariants(report)

    role_root = tmp_path / "role"
    plan, baseline, candidate = build_local_fixture(role_root)
    _rewrite_manifest(
        candidate, lambda value: value["comparison"].__setitem__("role", "baseline")
    )
    report = validate_pairing(plan, baseline, candidate, repository_root=role_root)
    assert report["exit_code"] == EXIT_VIOLATION
    assert "comparison_plan_binding_mismatch" in _invariants(report)


def test_damaged_generation_lineage_fails(tmp_path: Path) -> None:
    plan, baseline, candidate = build_local_fixture(tmp_path)
    manifest = RunManifestV1.model_validate_json(candidate.read_text(encoding="utf-8"))
    run_dir = tmp_path / manifest.output_directory
    store = BenchmarkRunCommitStore(run_dir)
    latest = store.load_latest().generation_path
    (latest / "COMMITTED").unlink()
    report = validate_pairing(plan, baseline, candidate, repository_root=tmp_path)
    assert report["exit_code"] == EXIT_VIOLATION
    assert "manifest_generation_state_mismatch" in _invariants(report)


def test_plan_binding_does_not_change_committed_semantic_records(tmp_path: Path) -> None:
    _, baseline, candidate = build_local_fixture(tmp_path)
    manifests = [
        RunManifestV1.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (baseline, candidate)
    ]
    states = [
        BenchmarkRunCommitStore(tmp_path / manifest.output_directory).load_latest()
        for manifest in manifests
    ]
    assert [row["normalized_result_identity_sha256"] for row in states[0].records] == [
        row["normalized_result_identity_sha256"] for row in states[1].records
    ]
    assert [row["semantic_events"] for row in states[0].records] == [
        row["semantic_events"] for row in states[1].records
    ]
    assert states[0].config["configuration"]["budgets"] == states[1].config[
        "configuration"
    ]["budgets"]


def test_frozen_and_registry_evidence_are_not_backfilled(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "profiles": [
                    {"profile_id": "record160"},
                    {"profile_id": "full1000"},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = audit_frozen_eligibility(legacy)
    assert report["exit_code"] == EXIT_NOT_READY
    assert all(row["status"] == "not_eligible" for row in report["frozen_profiles"])

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"strategies": [{"strategy_id": "legacy"}]}), encoding="utf-8")
    registry_report = audit_evidence_registry(registry)
    assert registry_report["exit_code"] == EXIT_NOT_READY
    assert registry_report["registry"]["evidence_conclusions_modified"] is False


def test_comparison_plan_serialization_is_deterministic(tmp_path: Path) -> None:
    plan_path, _, _ = build_local_fixture(tmp_path)
    first = plan_path.read_bytes()
    plan = load_comparison_plan(plan_path)
    plan_path.write_text(
        json.dumps(
            plan.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert plan_path.read_bytes() == first


def test_canonical_runner_requires_both_binding_arguments_and_exact_population(
    tmp_path: Path,
) -> None:
    plan_path, _, _ = build_local_fixture(tmp_path)
    plan = load_comparison_plan(plan_path)
    options = run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        run_id="pairing-fixture",
        comparison_plan_path=plan_path,
        comparison_role="baseline",
    )
    queries = [
        EvalQuery(query_id=value, query=f"offline query {index}")
        for index, value in enumerate(plan.queries.identities)
    ]
    run_benchmark._validate_comparison_population(options, queries)
    with pytest.raises(ValueError, match="population or order"):
        run_benchmark._validate_comparison_population(options, list(reversed(queries)))
    with pytest.raises(ValidationError):
        run_benchmark.BenchmarkRunOptions(
            dataset="auto_scholar_query",
            run_id="missing-role",
            comparison_plan_path=plan_path,
        )
