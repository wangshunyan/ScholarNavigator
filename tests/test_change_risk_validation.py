from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.change_risk_validation import (
    EXIT_INCOMPLETE,
    EXIT_SATISFIED,
    EXIT_VIOLATION,
    ValidationScopeError,
    ValidationScopeIncomplete,
    build_attestation,
    build_plan,
    canonical_json,
    load_protocol,
    protocol_template,
    stable_hash,
    verify_execution,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_validation_scope.py"
PROTOCOL_PATH = ROOT / "benchmark/change_risk_validation_v1_protocol.json"


def _freshness() -> dict[str, object]:
    return {
        "components": [
            {
                "component_id": "source_fusion",
                "files": [
                    "src/scholar_agent/evaluation/source_fusion_ablation.py",
                    "scripts/check_source_fusion_ablation.py",
                ],
            },
            {
                "component_id": "ranking_runtime",
                "files": ["src/scholar_agent/agents/reranker.py"],
            },
            {
                "component_id": "frontend_reproducible_build",
                "files": ["frontend/next.config.ts"],
            },
            {
                "component_id": "readiness_publication",
                "files": ["benchmark/global_protocol.json"],
            },
        ],
        "gates": [
            {
                "components": ["source_fusion"],
                "gate_id": "source_fusion",
            }
        ],
    }


def _readiness() -> dict[str, object]:
    return {
        "read_only_gates": [
            {
                "arguments": [
                    "scripts/check_source_fusion_ablation.py",
                    "verify",
                ],
                "expected_exit_code": 0,
                "gate_id": "source_fusion",
            }
        ]
    }


def _protocol() -> dict[str, object]:
    value = protocol_template()
    value["protocol_sha256"] = stable_hash(value)
    return value


def _plan(
    changes: list[dict[str, object]],
    *,
    target: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_plan(
        changes=changes,
        protocol=_protocol(),
        freshness=_freshness(),
        readiness=_readiness(),
        target=target
        or {
            "from_commit": "a" * 40,
            "mode": "worktree",
            "target_commit": "a" * 40,
            "worktree_sha256": "b" * 64,
        },
    )


def _executions(plan: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "command": row["command"],
            "exit_code": row["expected_exit_code"],
            "output_sha256": stable_hash(
                {"validation_id": row["validation_id"]}
            ),
            "validation_id": row["validation_id"],
        }
        for row in plan["required_validations"]
    ]


def test_document_only_plan_is_low_and_skips_full_and_frontend() -> None:
    plan = _plan(
        [{"old_path": None, "path": "docs/guide.md", "status": "M"}]
    )
    assert plan["overall_risk"] == "low"
    assert plan["full_pytest_required"] is False
    assert {row["validation_id"] for row in plan["explicit_skips"]} == {
        "frontend:build",
        "frontend:lint",
    }


def test_freshness_proven_semantic_noop_is_low_risk() -> None:
    plan = _plan(
        [
            {
                "old_path": None,
                "path": "src/scholar_agent/agents/reranker.py",
                "semantic_equivalent": True,
                "status": "M",
            }
        ]
    )
    assert plan["overall_risk"] == "low"
    assert plan["changes"][0]["reason"] == (
        "freshness_semantic_digest_unchanged"
    )
    assert plan["full_pytest_required"] is False


def test_isolated_component_selects_targeted_test_and_gate() -> None:
    plan = _plan(
        [
            {
                "old_path": None,
                "path": "src/scholar_agent/evaluation/source_fusion_ablation.py",
                "status": "M",
            }
        ]
    )
    assert plan["overall_risk"] == "targeted"
    ids = {row["validation_id"] for row in plan["required_validations"]}
    assert "pytest:tests/test_source_fusion_ablation.py" in ids
    assert "gate:source_fusion" in ids
    assert "pytest:full" not in ids


@pytest.mark.parametrize(
    "path",
    [
        "src/scholar_agent/services/search_service.py",
        "src/scholar_agent/agents/reranker.py",
        "benchmark/global_protocol.json",
    ],
)
def test_core_shared_and_protocol_changes_force_full_pytest(path: str) -> None:
    plan = _plan([{"old_path": None, "path": path, "status": "M"}])
    assert plan["overall_risk"] == "high"
    assert plan["full_pytest_required"] is True
    assert "pytest:full" in {
        row["validation_id"] for row in plan["required_validations"]
    }


@pytest.mark.parametrize(
    "path",
    [
        "frontend/src/types/api.ts",
        "frontend/next.config.ts",
        "src/scholar_agent/models/api.py",
    ],
)
def test_frontend_and_api_contract_changes_select_lint_and_build(path: str) -> None:
    plan = _plan([{"old_path": None, "path": path, "status": "M"}])
    ids = {row["validation_id"] for row in plan["required_validations"]}
    assert plan["overall_risk"] == "frontend"
    assert {"frontend:lint", "frontend:build"} <= ids


def test_rename_uses_old_component_and_cannot_evade_risk() -> None:
    plan = _plan(
        [
            {
                "old_path": "src/scholar_agent/agents/reranker.py",
                "path": "src/scholar_agent/evaluation/moved.py",
                "status": "R",
            }
        ]
    )
    assert plan["overall_risk"] == "high"
    assert plan["changes"][0]["components"] == ["ranking_runtime"]


def test_unregistered_semantic_and_untracked_file_fail_closed() -> None:
    with pytest.raises(
        ValidationScopeError, match="unregistered_semantic_change"
    ):
        _plan(
            [
                {
                    "old_path": None,
                    "path": "src/new_unregistered_module.py",
                    "status": "?",
                }
            ]
        )


def test_existing_third_party_state_is_ignored_without_hiding_other_changes() -> None:
    plan = _plan(
        [
            {
                "old_path": None,
                "path": "third_party/paper-qa",
                "status": "m",
            },
            {
                "old_path": None,
                "path": "src/scholar_agent/services/search_service.py",
                "status": "M",
            },
        ]
    )
    ignored = next(
        row for row in plan["changes"] if row["path"] == "third_party/paper-qa"
    )
    assert ignored["validation_relevant"] is False
    assert plan["full_pytest_required"] is True


def test_other_third_party_and_env_changes_are_not_silently_ignored() -> None:
    with pytest.raises(ValidationScopeError, match="unregistered_change"):
        _plan(
            [
                {
                    "old_path": None,
                    "path": "third_party/other-vendor",
                    "status": "m",
                }
            ]
        )
    with pytest.raises(ValidationScopeError, match="forbidden_env_path"):
        _plan(
            [
                {
                    "old_path": None,
                    "path": ".env",
                    "status": "M",
                }
            ]
        )


def test_execution_rejects_deleted_command_and_forged_pass() -> None:
    plan = _plan(
        [
            {
                "old_path": None,
                "path": "src/scholar_agent/agents/reranker.py",
                "status": "M",
            }
        ]
    )
    executions = _executions(plan)
    missing = build_attestation(
        plan,
        executions=executions[:-1],
        executed_head="a" * 40,
        final_head="a" * 40,
        tested_worktree_sha256="b" * 64,
    )
    with pytest.raises(ValidationScopeIncomplete):
        verify_execution(plan, missing)
    failed_rows = copy.deepcopy(executions)
    failed_rows[0]["exit_code"] = 1
    failed = build_attestation(
        plan,
        executions=failed_rows,
        executed_head="a" * 40,
        final_head="a" * 40,
        tested_worktree_sha256="b" * 64,
    )
    with pytest.raises(ValidationScopeError, match="validation_execution_failed"):
        verify_execution(plan, failed)


def test_final_head_drift_requires_full_pytest() -> None:
    targeted = _plan(
        [
            {
                "old_path": None,
                "path": "src/scholar_agent/evaluation/source_fusion_ablation.py",
                "status": "M",
            }
        ]
    )
    attestation = build_attestation(
        targeted,
        executions=_executions(targeted),
        executed_head="a" * 40,
        final_head="c" * 40,
        tested_worktree_sha256="b" * 64,
    )
    with pytest.raises(
        ValidationScopeIncomplete, match="final_head_drift_requires_full_pytest"
    ):
        verify_execution(targeted, attestation)
    high = _plan(
        [
            {
                "old_path": None,
                "path": "src/scholar_agent/services/search_service.py",
                "status": "M",
            }
        ]
    )
    high_attestation = build_attestation(
        high,
        executions=_executions(high),
        executed_head="a" * 40,
        final_head="c" * 40,
        tested_worktree_sha256="b" * 64,
    )
    assert verify_execution(high, high_attestation)["exit_code"] == 0


def test_changed_test_must_be_executed() -> None:
    plan = _plan(
        [
            {
                "old_path": None,
                "path": "tests/test_change_risk_validation.py",
                "status": "M",
            }
        ]
    )
    assert "pytest:tests/test_change_risk_validation.py" in {
        row["validation_id"] for row in plan["required_validations"]
    }


def test_plan_and_attestation_are_byte_deterministic() -> None:
    changes = [
        {
            "old_path": None,
            "path": "src/scholar_agent/services/search_service.py",
            "status": "M",
        }
    ]
    first = _plan(changes)
    second = _plan(list(reversed(changes)))
    assert canonical_json(first) == canonical_json(second)
    first_attestation = build_attestation(
        first,
        executions=_executions(first),
        executed_head="a" * 40,
        final_head="a" * 40,
        tested_worktree_sha256="b" * 64,
    )
    second_attestation = build_attestation(
        second,
        executions=_executions(second),
        executed_head="a" * 40,
        final_head="a" * 40,
        tested_worktree_sha256="b" * 64,
    )
    assert canonical_json(first_attestation) == canonical_json(second_attestation)


def test_protocol_drift_and_cli_missing_evidence_fail_closed(tmp_path: Path) -> None:
    load_protocol(PROTOCOL_PATH)
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    value["component_risk"]["default_registered_component"] = "low"
    payload = copy.deepcopy(value)
    payload["protocol_sha256"] = "0" * 64
    value["protocol_sha256"] = stable_hash(payload)
    changed = tmp_path / "protocol.json"
    changed.write_bytes(canonical_json(value))
    with pytest.raises(ValidationScopeError, match="protocol_schema_invalid"):
        load_protocol(changed)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify-execution",
            "--plan",
            str(tmp_path / "missing-plan.json"),
            "--attestation",
            str(tmp_path / "missing-attestation.json"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == EXIT_VIOLATION
    assert completed.stderr == b""
    assert b"Traceback" not in completed.stdout


def test_cli_audit_is_deterministic() -> None:
    command = [sys.executable, str(SCRIPT), "audit-current"]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert first.returncode == second.returncode == EXIT_SATISFIED
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
