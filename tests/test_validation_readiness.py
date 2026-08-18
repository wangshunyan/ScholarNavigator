from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.validation_readiness import (
    ValidationReadinessError,
    ValidationReadinessNotReady,
    build_release_files,
    canonical_json,
    load_contract,
    sha256_file,
    stable_tree_sha256,
    verify_release_files,
    write_release_files,
    _run_read_only_gates,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for name in ("README.md", "architecture.md", "evaluation.md", "contest.md"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    for name in ("generator.py", "cli.py", "guide.md"):
        (root / name).write_text(f"{name}\n", encoding="utf-8")

    evidence_values = {
        "record.json": {
            "implementation_base_commit": "base-commit",
            "record": 162,
            "main": 160,
            "snapshot": 1093,
            "final": 1396,
        },
        "full.json": {"status": "incomplete"},
        "human.json": {"status": "pending"},
        "official.json": {"status": "blocked"},
        "policy.json": {"current": ["current_rules"], "v2": False},
    }
    for name, value in evidence_values.items():
        _write_json(root / name, value)

    def evidence(
        evidence_id: str,
        path: str,
        *,
        checks: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "checks": checks or [],
            "dependencies": [],
            "evidence_id": evidence_id,
            "format": "json",
            "path": path,
            "protocol": f"{evidence_id}_v1",
            "required": True,
            "role": "aggregate",
            "sha256": sha256_file(root / path),
        }

    contract: dict[str, object] = {
        "schema_version": "1",
        "protocol": "validation_readiness_bundle_v1",
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "code_identity": {
            "implementation_base_commit": "base-commit",
            "files": ["contract.json", "generator.py", "cli.py", "guide.md"],
        },
        "workspace": {
            "preserved_path": "third_party/paper-qa",
            "required_state": "modified_nested_worktree",
        },
        "claim_sources": [
            {
                "document_id": key,
                "path": path,
                "sha256": sha256_file(root / path),
            }
            for key, path in (
                ("readme", "README.md"),
                ("architecture", "architecture.md"),
                ("evaluation", "evaluation.md"),
                ("contest", "contest.md"),
            )
        ],
        "evidence": [
            evidence(
                "record",
                "record.json",
                checks=[
                    {"pointer": "/implementation_base_commit", "equals": "base-commit"}
                ],
            ),
            evidence("full", "full.json", checks=[{"pointer": "/status", "equals": "incomplete"}]),
            evidence("human", "human.json", checks=[{"pointer": "/status", "equals": "pending"}]),
            evidence("official", "official.json", checks=[{"pointer": "/status", "equals": "blocked"}]),
            evidence("policy", "policy.json", checks=[{"pointer": "/v2", "equals": False}]),
        ],
        "consistency_assertions": [
            {
                "assertion_id": "record_count",
                "expected": 162,
                "observations": [{"evidence_id": "record", "pointer": "/record"}],
            },
            {
                "assertion_id": "v2_disabled",
                "expected": False,
                "observations": [{"evidence_id": "policy", "pointer": "/v2"}],
            },
        ],
        "blockers": [
            {
                "blocker_id": "full1000_incomplete",
                "status": "blocked",
                "evidence_ids": ["full"],
                "missing_external_input": "complete run",
                "future_integration_point": "new immutable run",
                "non_substitutes": ["partial run"],
            },
            {
                "blocker_id": "human_precision_missing",
                "status": "blocked",
                "evidence_ids": ["human"],
                "missing_external_input": "human labels",
                "future_integration_point": "label import",
                "non_substitutes": ["proxy labels"],
            },
            {
                "blocker_id": "official_scorer_schema_missing",
                "status": "blocked",
                "evidence_ids": ["official"],
                "missing_external_input": "official scorer",
                "future_integration_point": "official adapter",
                "non_substitutes": ["internal metric"],
            },
        ],
        "claims": [
            {
                "claim_id": "readme_scope",
                "document_id": "readme",
                "statement": "Operational instructions only.",
                "status": "not_applicable",
                "scope": "operational_documentation",
                "evidence_ids": ["policy"],
                "boundary": "No quality statement.",
            },
            {
                "claim_id": "architecture_gate",
                "document_id": "architecture",
                "statement": "Offline evidence integrity is implemented.",
                "status": "verified",
                "scope": "engineering_capability",
                "evidence_ids": ["record"],
                "boundary": "Engineering only.",
            },
            {
                "claim_id": "evaluation_internal",
                "document_id": "evaluation",
                "statement": "The partial run is internal evidence.",
                "status": "internal_only",
                "scope": "internal_validation",
                "evidence_ids": ["record"],
                "boundary": "Not a formal result.",
            },
            {
                "claim_id": "human_blocked",
                "document_id": "evaluation",
                "statement": "Human validation is incomplete.",
                "status": "blocked",
                "scope": "formal_validation_requirement",
                "blocker_id": "human_precision_missing",
                "evidence_ids": ["human"],
                "boundary": "No proxy substitution.",
            },
            {
                "claim_id": "full_blocked",
                "document_id": "contest",
                "statement": "The full run is incomplete.",
                "status": "blocked",
                "scope": "formal_validation_requirement",
                "blocker_id": "full1000_incomplete",
                "evidence_ids": ["full"],
                "boundary": "Partial is not full.",
            },
            {
                "claim_id": "official_blocked",
                "document_id": "contest",
                "statement": "Official alignment is unavailable.",
                "status": "blocked",
                "scope": "formal_validation_requirement",
                "blocker_id": "official_scorer_schema_missing",
                "evidence_ids": ["official"],
                "boundary": "Internal is not official.",
            },
        ],
        "read_only_gates": [],
        "release": {
            "status": "ready_with_declared_blockers",
            "generation_command": "python scripts/check.py generate",
            "verification_command": "python scripts/check.py verify",
        },
    }
    _write_json(root / "contract.json", contract)
    return root, contract


def _save_contract(root: Path, contract: dict[str, object]) -> dict[str, object]:
    _write_json(root / "contract.json", contract)
    return load_contract(root / "contract.json")


def _rehash(contract: dict[str, object], root: Path, evidence_id: str) -> None:
    for item in contract["evidence"]:  # type: ignore[index]
        if item["evidence_id"] == evidence_id:
            item["sha256"] = sha256_file(root / str(item["path"]))
            return
    raise AssertionError(evidence_id)


def _build(root: Path, contract: dict[str, object]) -> dict[str, bytes]:
    return build_release_files(
        contract,
        repository_root=root,
        run_gates=False,
        check_workspace=False,
    )


def test_release_is_byte_deterministic_and_verifiable(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    contract = _save_contract(root, contract)
    first = _build(root, contract)
    second = _build(root, contract)
    assert first == second
    assert stable_tree_sha256(first) == stable_tree_sha256(second)

    output = root / "release"
    assert write_release_files(output, first) == stable_tree_sha256(first)
    report = verify_release_files(
        contract,
        output,
        repository_root=root,
        run_gates=False,
        check_workspace=False,
    )
    assert report["status"] == "ready_with_declared_blockers"
    assert report["claim_evidence_coverage_rate"] == 1.0
    assert report["blocker_count"] == 3


def test_read_only_gate_expands_temporary_output_without_literal_workspace_leak(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    literal = root / "{temporary_output}"
    contract = {
        "read_only_gates": [
            {
                "arguments": [
                    "-c",
                    (
                        "import json,pathlib,sys; "
                        "path=pathlib.Path(sys.argv[1]); "
                        "(path/'report.json').write_text('{}'); "
                        "print(json.dumps({'status':'ok'}))"
                    ),
                    "{temporary_output}",
                ],
                "checks": [{"pointer": "/status", "equals": "ok"}],
                "expected_exit_code": 0,
                "gate_id": "temporary_output_expansion",
                "timeout_seconds": 10,
            }
        ]
    }
    assert _run_read_only_gates(contract, root)[0]["gate_id"] == "temporary_output_expansion"
    assert not literal.exists()


def test_missing_and_tampered_evidence_are_distinct(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    contract = _save_contract(root, contract)
    (root / "record.json").unlink()
    with pytest.raises(ValidationReadinessNotReady, match="required_evidence_missing"):
        _build(root, contract)

    root, contract = _fixture(tmp_path / "tamper")
    contract = _save_contract(root, contract)
    (root / "record.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationReadinessError, match="historical_evidence_hash_drift"):
        _build(root, contract)


def test_cross_report_count_conflict_fails_after_valid_rehash(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    record = json.loads((root / "record.json").read_text(encoding="utf-8"))
    record["record"] = 161
    _write_json(root / "record.json", record)
    _rehash(contract, root, "record")
    contract = _save_contract(root, contract)
    with pytest.raises(
        ValidationReadinessError, match="cross_evidence_count_or_state_conflict"
    ):
        _build(root, contract)


def test_formal_requirement_cannot_be_overclaimed(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    claims = contract["claims"]  # type: ignore[index]
    next(item for item in claims if item["claim_id"] == "full_blocked")["status"] = "verified"
    contract = _save_contract(root, contract)
    with pytest.raises(ValidationReadinessError, match="formal_requirement_overclaim"):
        _build(root, contract)


def test_required_blocker_omission_is_rejected(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    contract["blockers"] = contract["blockers"][:-1]  # type: ignore[index]
    _write_json(root / "contract.json", contract)
    with pytest.raises(ValidationReadinessError, match="required_blocker_set_drift"):
        load_contract(root / "contract.json")


def test_old_commit_evidence_mixing_is_rejected(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    record = json.loads((root / "record.json").read_text(encoding="utf-8"))
    record["implementation_base_commit"] = "other-commit"
    _write_json(root / "record.json", record)
    _rehash(contract, root, "record")
    contract = _save_contract(root, contract)
    with pytest.raises(ValidationReadinessError, match="evidence_field_drift"):
        _build(root, contract)


def test_default_enabled_v2_is_rejected(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    policy = json.loads((root / "policy.json").read_text(encoding="utf-8"))
    policy["v2"] = True
    _write_json(root / "policy.json", policy)
    _rehash(contract, root, "policy")
    contract = _save_contract(root, contract)
    with pytest.raises(ValidationReadinessError, match="evidence_field_drift"):
        _build(root, contract)


def test_absolute_path_leak_is_rejected(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    contract["claims"][0]["statement"] = "/Users/example/private/output"  # type: ignore[index]
    _write_json(root / "contract.json", contract)
    with pytest.raises(ValidationReadinessError, match="absolute_path_leak"):
        load_contract(root / "contract.json")


def test_release_member_tampering_is_rejected(tmp_path: Path) -> None:
    root, contract = _fixture(tmp_path)
    contract = _save_contract(root, contract)
    output = root / "release"
    write_release_files(output, _build(root, contract))
    (output / "claims.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationReadinessError, match="release_file_byte_drift"):
        verify_release_files(
            contract,
            output,
            repository_root=root,
            run_gates=False,
            check_workspace=False,
        )
