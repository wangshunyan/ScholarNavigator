from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.check_formal_evidence_quarantine as cli
from scholar_agent.evaluation.formal_evidence_quarantine import (
    EXIT_BLOCKED,
    EXIT_READY,
    EXIT_VIOLATION,
    QuarantineError,
    audit_contamination,
    canonical_json,
    consume_for_evaluation,
    current_readiness,
    load_protocol,
    quarantine_io_guard,
    stable_hash,
    synthetic_manifest,
    verify_boundaries,
    verify_intake_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_evidence_quarantine_v1_protocol.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH)


def _fixture_protocol(protocol: dict[str, object], roots: list[str]) -> dict[str, object]:
    value = copy.deepcopy(protocol)
    value["forbidden_consumer_roots"] = roots
    return value


def test_current_boundaries_and_readiness_are_closed(protocol: dict[str, object]) -> None:
    first = verify_boundaries(ROOT, protocol)
    second = verify_boundaries(ROOT, protocol)
    assert first["status"] == "passed"
    assert first["violation_count"] == 0
    assert canonical_json(first) == canonical_json(second)
    readiness = current_readiness(ROOT, protocol)
    assert readiness["exit_code"] == EXIT_BLOCKED
    assert readiness["controls_ready"] is True
    assert readiness["formal_validation_complete"] is False


def test_direct_and_indirect_production_imports_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    retrieval = tmp_path / "src/scholar_agent/retrieval"
    core = tmp_path / "src/scholar_agent/core"
    retrieval.mkdir(parents=True)
    core.mkdir(parents=True)
    (retrieval / "direct.py").write_text(
        "from scholar_agent.evaluation.formal_evidence_quarantine import load_intake_manifest\n",
        encoding="utf-8",
    )
    (retrieval / "indirect.py").write_text(
        "from scholar_agent.core.formal_bridge import read\n", encoding="utf-8"
    )
    (core / "formal_bridge.py").write_text(
        "import scholar_agent.evaluation.formal_evidence_quarantine\n", encoding="utf-8"
    )
    report = verify_boundaries(
        tmp_path,
        _fixture_protocol(
            protocol, ["src/scholar_agent/retrieval", "src/scholar_agent/core"]
        ),
    )
    assert report["exit_code"] == EXIT_VIOLATION
    assert report["violation_count"] == 3
    assert any(row["path"].endswith("retrieval/indirect.py") for row in report["violations"])


def test_package_init_and_from_package_submodule_bridges_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    retrieval = tmp_path / "src/scholar_agent/retrieval"
    core = tmp_path / "src/scholar_agent/core"
    retrieval.mkdir(parents=True)
    core.mkdir(parents=True)
    (core / "formal_bridge.py").write_text(
        "import scholar_agent.evaluation.formal_evidence_quarantine\n", encoding="utf-8"
    )
    (retrieval / "absolute.py").write_text(
        "from scholar_agent.core import formal_bridge\n", encoding="utf-8"
    )
    (retrieval / "formal_bridge.py").write_text(
        "import scholar_agent.evaluation.formal_evidence_quarantine\n", encoding="utf-8"
    )
    (retrieval / "relative.py").write_text(
        "from . import formal_bridge\n", encoding="utf-8"
    )
    (retrieval / "__init__.py").write_text(
        "from . import formal_bridge\n", encoding="utf-8"
    )
    report = verify_boundaries(
        tmp_path,
        _fixture_protocol(
            protocol, ["src/scholar_agent/retrieval", "src/scholar_agent/core"]
        ),
    )
    paths = {row["path"] for row in report["violations"]}
    assert report["exit_code"] == EXIT_VIOLATION
    assert "src/scholar_agent/retrieval/absolute.py" in paths
    assert "src/scholar_agent/retrieval/relative.py" in paths
    assert "src/scholar_agent/retrieval/__init__.py" in paths


def test_existing_services_selection_dependency_is_not_a_false_positive(
    protocol: dict[str, object],
) -> None:
    report = verify_boundaries(ROOT, protocol)
    assert report["exit_code"] == EXIT_READY
    assert not any(row["path"].startswith("src/scholar_agent/services/") for row in report["violations"])


def test_relative_import_and_frontend_asset_leakage_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    retrieval = tmp_path / "src/scholar_agent/retrieval"
    frontend = tmp_path / "frontend/src"
    retrieval.mkdir(parents=True)
    frontend.mkdir(parents=True)
    (retrieval / "relative.py").write_text(
        "from ..evaluation.formal_evidence_quarantine import consume_for_evaluation\n",
        encoding="utf-8",
    )
    (frontend / "search.ts").write_text(
        "export const forbidden = 'official_scorer_output';\n", encoding="utf-8"
    )
    report = verify_boundaries(
        tmp_path,
        _fixture_protocol(
            protocol, ["src/scholar_agent/retrieval", "frontend/src"]
        ),
    )
    assert report["exit_code"] == EXIT_VIOLATION
    assert {row["invariant"] for row in report["violations"]} == {
        "production_imports_formal_evidence",
        "formal_evidence_token_in_runtime_asset",
    }


@pytest.mark.parametrize(
    "source",
    [
        "open('formal_evidence/labels.json').read()\n",
        "PROMPT_CONFIG['qrels'] = 'inject'\n",
        "CACHE['official_scorer_output'] = 'reuse'\n",
        "EVENT = {'gold': 'leak'}\n",
    ],
)
def test_file_prompt_and_configuration_leakage_are_rejected(
    tmp_path: Path, protocol: dict[str, object], source: str
) -> None:
    target = tmp_path / "src/scholar_agent/prompts"
    target.mkdir(parents=True)
    (target / "unsafe.py").write_text(source, encoding="utf-8")
    report = verify_boundaries(
        tmp_path, _fixture_protocol(protocol, ["src/scholar_agent/prompts"])
    )
    assert report["exit_code"] == EXIT_VIOLATION
    assert report["violation_count"] >= 1


@pytest.mark.parametrize("evidence_type", ["human_annotation_labels", "official_scorer_output"])
def test_legal_evaluation_only_consumption_and_runtime_copy_denial(
    tmp_path: Path, protocol: dict[str, object], evidence_type: str
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol, evidence_type=evidence_type)
    expected = (tmp_path / manifest.artifact.path).read_bytes()
    actual = consume_for_evaluation(
        manifest,
        evidence_root=tmp_path,
        consumer="scholar_agent.evaluation.offline_report",
        purpose="reporting",
        protocol=protocol,
    )
    assert actual == expected
    with pytest.raises(QuarantineError, match="consumer_not_allowed"):
        consume_for_evaluation(
            manifest,
            evidence_root=tmp_path,
            consumer="scholar_agent.services.search_service",
            purpose="evaluation",
            protocol=protocol,
        )
    artifact = tmp_path / manifest.artifact.path
    with quarantine_io_guard(artifact=artifact):
        with pytest.raises(QuarantineError, match="copy_denied"):
            shutil.copy2(artifact, tmp_path / "copied.json")


def test_artifact_tamper_and_cross_protocol_mix_are_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol)
    (tmp_path / manifest.artifact.path).write_text("tampered", encoding="utf-8")
    with pytest.raises(QuarantineError, match="hash_mismatch"):
        verify_intake_manifest(manifest, evidence_root=tmp_path, protocol=protocol)

    clean_root = tmp_path / "clean"
    manifest = synthetic_manifest(clean_root, protocol)
    changed = copy.deepcopy(protocol)
    changed["allowed_consumer_prefixes"] = ["scholar_agent.evaluation.other."]
    with pytest.raises(QuarantineError, match="consumer_drift"):
        verify_intake_manifest(manifest, evidence_root=clean_root, protocol=changed)


def test_unlocked_lifecycle_cannot_be_consumed(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol)
    payload = manifest.model_dump(mode="json")
    payload["lifecycle_state"] = "received"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = stable_hash(payload)
    path = tmp_path / "received.json"
    path.write_bytes(canonical_json(payload))
    from scholar_agent.evaluation.formal_evidence_quarantine import load_intake_manifest

    received = load_intake_manifest(path)
    with pytest.raises(QuarantineError, match="lifecycle_not_consumable"):
        consume_for_evaluation(
            received,
            evidence_root=tmp_path,
            consumer="scholar_agent.evaluation.offline_report",
            purpose="reporting",
            protocol=protocol,
        )


@pytest.mark.parametrize(
    ("path", "component"),
    [
        ("src/scholar_agent/agents/reranker.py", "ranking"),
        ("src/scholar_agent/prompts/manifest.json", "prompt"),
        ("src/scholar_agent/core/config.py", "default_policy"),
        ("src/scholar_agent/retrieval/query_adapter.py", "query_planning"),
    ],
)
def test_posthoc_changes_make_formal_claim_stale(
    tmp_path: Path, protocol: dict[str, object], path: str, component: str
) -> None:
    manifest = synthetic_manifest(tmp_path / component, protocol)
    report = audit_contamination(manifest, [path], protocol)
    assert report["status"] == "stale_for_claim"
    assert report["exit_code"] == EXIT_VIOLATION
    assert component in report["affected_components"]


def test_reporting_only_change_does_not_reselect_strategy(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol)
    report = audit_contamination(manifest, ["docs/formal-evaluation-report.md"], protocol)
    assert report["status"] == "clean"
    assert report["minimum_rerun_components"] == []


def test_synthetic_chronology_cannot_claim_real_evidence(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol)
    payload = manifest.model_dump(mode="json")
    payload["synthetic_only"] = False
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = stable_hash(payload)
    path = tmp_path / "spoofed.json"
    path.write_bytes(canonical_json(payload))
    from scholar_agent.evaluation.formal_evidence_quarantine import load_intake_manifest

    spoofed = load_intake_manifest(path)
    with pytest.raises(QuarantineError, match="synthetic_chronology"):
        verify_intake_manifest(spoofed, evidence_root=tmp_path, protocol=protocol)


def test_cross_commit_chronology_is_rejected(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    manifest = synthetic_manifest(tmp_path, protocol)
    payload = manifest.model_dump(mode="json")
    payload["synthetic_only"] = False
    payload["chronology"] = {
        "preregistration_commit": "f" * 40,
        "execution_commit": "e1e2545cab9d6f2ecf0e95be157d4c1e71376ec8",
        "intake_commit": protocol["source_commit"],
        "report_code_commit": "6e3b5518b6b58c8384072a4e5a99ac1b6e649712",
        "proof": "git_ancestry",
    }
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = stable_hash(payload)
    path = tmp_path / "cross-commit.json"
    path.write_bytes(canonical_json(payload))
    from scholar_agent.evaluation.formal_evidence_quarantine import load_intake_manifest

    mixed = load_intake_manifest(path)
    with pytest.raises(QuarantineError, match="chronology_invalid"):
        verify_intake_manifest(
            mixed, evidence_root=tmp_path, protocol=protocol, repository_root=ROOT
        )


def test_cli_readiness_and_determinism(
    protocol: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["audit-readiness"]) == EXIT_BLOCKED
    first = capsys.readouterr().out
    assert cli.main(["audit-readiness"]) == EXIT_BLOCKED
    second = capsys.readouterr().out
    assert first == second
    assert '"formal_validation_complete": false' in first


def test_cli_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 4
    assert '"status": "usage_error"' in capsys.readouterr().out


def test_cli_missing_artifact_is_stable_exit_two_without_traceback(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    binding = tmp_path / "binding.json"
    chronology = tmp_path / "chronology.json"
    binding.write_text(
        json.dumps(
            {
                "contract": "comparison_plan_v1",
                "plan_sha256": digest,
                "run_manifest_sha256": digest,
                "query_order_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    chronology.write_text(
        json.dumps(
            {
                "preregistration_commit": "1" * 40,
                "execution_commit": "2" * 40,
                "intake_commit": "3" * 40,
                "report_code_commit": "4" * 40,
                "proof": "synthetic_fixture_only",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_formal_evidence_quarantine.py"),
            "intake-dry-run",
            "--artifact",
            str(tmp_path / "missing.json"),
            "--evidence-root",
            str(tmp_path),
            "--evidence-type",
            "human_annotation_labels",
            "--binding",
            str(binding),
            "--chronology",
            str(chronology),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == EXIT_VIOLATION
    assert report["error_code"] == "evidence_artifact_unavailable"
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout


@pytest.mark.parametrize(
    "kind",
    [
        "missing_protocol",
        "malformed_protocol",
        "incomplete_protocol",
        "malformed_binding",
    ],
)
def test_cli_bad_protocol_or_json_never_leaks_traceback(
    tmp_path: Path, kind: str
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{broken", encoding="utf-8")
    if kind == "missing_protocol":
        protocol_path = tmp_path / "absent.json"
    elif kind == "incomplete_protocol":
        incomplete = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        incomplete.pop("allowed_consumer_prefixes")
        protocol_path.write_text(json.dumps(incomplete), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts/check_formal_evidence_quarantine.py"),
        "--protocol",
        str(protocol_path if kind != "malformed_binding" else PROTOCOL_PATH),
    ]
    if kind == "malformed_binding":
        bad = tmp_path / "binding.json"
        bad.write_text("[]", encoding="utf-8")
        chronology = tmp_path / "chronology.json"
        chronology.write_text("{}", encoding="utf-8")
        artifact = tmp_path / "artifact.json"
        artifact.write_text("{}", encoding="utf-8")
        command.extend(
            [
                "intake-dry-run",
                "--artifact",
                str(artifact),
                "--evidence-root",
                str(tmp_path),
                "--evidence-type",
                "human_annotation_labels",
                "--binding",
                str(bad),
                "--chronology",
                str(chronology),
            ]
        )
    else:
        command.append("verify-boundaries")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    report = json.loads(completed.stdout)
    assert completed.returncode == EXIT_VIOLATION
    assert report["exit_code"] == EXIT_VIOLATION
    if kind == "incomplete_protocol":
        assert report["error_code"] == "protocol_schema_invalid"
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout


@pytest.mark.parametrize(
    "mutation",
    [
        "allowed_consumer",
        "empty_forbidden_roots",
        "replace_forbidden_root",
        "delete_prohibited_use",
        "source_commit",
    ],
)
def test_cli_rejects_frozen_security_policy_drift_without_traceback(
    tmp_path: Path, mutation: str
) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if mutation == "allowed_consumer":
        payload["allowed_consumer_prefixes"] = ["scholar_agent.services."]
    elif mutation == "empty_forbidden_roots":
        payload["forbidden_consumer_roots"] = []
    elif mutation == "replace_forbidden_root":
        payload["forbidden_consumer_roots"][0] = "src/nonexistent"
    elif mutation == "delete_prohibited_use":
        payload["prohibited_uses"].pop()
    else:
        payload["source_commit"] = "f" * 40
    protocol_path = tmp_path / f"{mutation}.json"
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_formal_evidence_quarantine.py"),
            "--protocol",
            str(protocol_path),
            "verify-boundaries",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == EXIT_VIOLATION
    assert report["error_code"] == "protocol_schema_invalid"
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout
