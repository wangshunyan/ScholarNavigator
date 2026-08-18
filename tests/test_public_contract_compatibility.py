from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import scholar_agent.evaluation.public_contract_compatibility as contract_module
from scholar_agent.core.api_schemas import CostReport, HealthResponse

from scholar_agent.evaluation.public_contract_compatibility import (
    ContractError,
    build_snapshot,
    canonical_json,
    compare_snapshots,
    load_protocol,
    load_json,
    migrate_snapshot,
    recursive_schema,
    stable_hash,
    validate_snapshot,
    verify_current,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
BASELINE_PATH = ROOT / "benchmark/public_contract_compatibility_v1_baseline.json"
CLI = ROOT / "scripts/check_public_contract_compatibility.py"


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["content_sha256"] = stable_hash({key: child for key, child in value.items() if key != "content_sha256"})
    return value


@lru_cache(maxsize=1)
def _cached_snapshot() -> dict[str, object]:
    return build_snapshot(load_protocol(PROTOCOL_PATH), repository_root=ROOT)


def _snapshot() -> dict[str, object]:
    return copy.deepcopy(_cached_snapshot())


def test_current_snapshot_is_deterministic_and_matches_baseline() -> None:
    first = _snapshot()
    second = _snapshot()
    assert canonical_json(first) == canonical_json(second)
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert verify_current(load_protocol(PROTOCOL_PATH), baseline, repository_root=ROOT)["status"] == "contracts_compatible"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value["openapi"]["paths"].pop(next(iter(value["openapi"]["paths"]))), "field_removed"),
        (lambda value: value["openapi"].__setitem__("paths", []), "type_changed"),
        (lambda value: value["cli"]["run_provenance"].__setitem__("exit_codes", [0, 2, 3]), "ordered_or_membership_changed"),
        (lambda value: value["frontend"]["declarations"].pop("HealthResponse"), "field_removed"),
    ],
)
def test_breaking_contract_changes_are_rejected(mutate, reason: str) -> None:
    baseline = _snapshot()
    changed = copy.deepcopy(baseline)
    mutate(changed)
    _rehash(changed)
    report = compare_snapshots(baseline, changed)
    assert report["classification"] == "breaking"
    assert reason in {row["reason"] for row in report["changes"]}


def test_enum_narrowing_is_breaking() -> None:
    baseline = _snapshot()
    baseline["openapi"]["synthetic_enum_contract"] = {"enum": ["alpha", "beta"]}
    _rehash(baseline)
    changed = copy.deepcopy(baseline)
    schema = changed["openapi"]["synthetic_enum_contract"]
    schema["enum"] = schema["enum"][:-1]
    _rehash(changed)
    report = compare_snapshots(baseline, changed)
    assert report["classification"] == "breaking"
    assert any(row["reason"] == "enum_narrowed" for row in report["changes"])


def test_optional_addition_requires_explicit_review_policy() -> None:
    baseline = _snapshot()
    changed = copy.deepcopy(baseline)
    run_manifest = changed["artifacts"]["run_manifest_v1"]["json_schema"]
    assert run_manifest["additionalProperties"] is False
    run_manifest["properties"]["synthetic_optional"] = {"type": "string"}
    _rehash(changed)
    assert compare_snapshots(baseline, changed)["classification"] == "breaking"
    assert compare_snapshots(baseline, changed, extension_policy="allow_optional")["classification"] == "breaking"

    extensible = copy.deepcopy(baseline)
    extensible["openapi"]["components"]["schemas"]["SyntheticExtensible"] = {
        "additionalProperties": True,
        "properties": {},
        "required": [],
        "type": "object",
    }
    pointer = "/openapi/components/schemas/SyntheticExtensible/properties"
    extensible["extension_policies"][pointer] = "allow_optional"
    _rehash(extensible)
    additive = copy.deepcopy(extensible)
    additive["openapi"]["components"]["schemas"]["SyntheticExtensible"]["properties"]["note"] = {"type": "string"}
    _rehash(additive)
    assert compare_snapshots(extensible, additive)["classification"] == "additive_review_required"


def test_cli_has_no_global_extension_policy_override() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "compare",
            "--from",
            str(BASELINE_PATH),
            "--to",
            str(BASELINE_PATH),
            "--allow-optional-additions",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 4
    assert result.stderr == ""


def test_tampered_baseline_and_same_version_drift_fail_closed() -> None:
    baseline = _snapshot()
    baseline["cli"]["run_provenance"]["commands"].append("silent-drift")
    with pytest.raises(ContractError, match="contract_snapshot_hash_mismatch"):
        validate_snapshot(baseline)


def test_frontend_openapi_directional_required_nullable_and_type_contracts() -> None:
    openapi = contract_module._openapi_contract()
    frontend = contract_module._extract_ts_interface_contracts(ROOT / "frontend/src/types/api.ts")
    aliases = contract_module._extract_ts_alias_types(ROOT / "frontend/src/types/api.ts")
    report = contract_module._frontend_openapi_consistency(openapi, frontend, aliases)
    health = report["models"]["HealthResponse"]["fields"]["status"]
    assert health["backend_schema_required"] is False
    assert health["frontend_required"] is True
    assert health["wire_present_on_response"] is True
    assert "status" in report["response_serialization_fixtures"]["HealthResponse"]["present_fields"]
    assert set(report["response_serialization_fixtures"]["CostReport"]["present_fields"]) == set(
        openapi["components"]["schemas"]["CostReport"]["properties"]
    )

    optional_request = copy.deepcopy(frontend)
    optional_request["SearchRunCreateRequest"]["query"]["required"] = False
    with pytest.raises(ContractError, match="frontend_request_required_mismatch"):
        contract_module._frontend_openapi_consistency(openapi, optional_request, aliases)

    nullable_response = copy.deepcopy(openapi)
    nullable_response["components"]["schemas"]["HealthResponse"]["properties"]["status"] = {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    with pytest.raises(ContractError, match="frontend_response_nullability_mismatch"):
        contract_module._frontend_openapi_consistency(nullable_response, frontend, aliases)

    wrong_type = copy.deepcopy(frontend)
    wrong_type["HealthResponse"]["status"]["type"] = "boolean"
    with pytest.raises(ContractError, match="frontend_openapi_type_mismatch"):
        contract_module._frontend_openapi_consistency(openapi, wrong_type, aliases)


def test_real_fastapi_response_serialization_includes_defaulted_output_fields() -> None:
    app = FastAPI()

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, object]:
        return {"version": "fixture", "time": "2000-01-01T00:00:00Z"}

    @app.get("/cost", response_model=CostReport)
    def cost() -> dict[str, object]:
        return {}

    client = TestClient(app)
    assert set(client.get("/health").json()) == {"status", "time", "version"}
    assert set(client.get("/cost").json()) == set(CostReport.model_fields)


def test_cli_executable_probe_detects_exit_and_machine_schema_drift(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text('import json; print(json.dumps({"status": "ok", "nested": {"value": 1}})); raise SystemExit(3)\n', encoding="utf-8")
    probe = {
        "arguments": [],
        "command": "synthetic",
        "expected_exit_code": 3,
        "probe_id": "synthetic",
    }
    first = contract_module._single_cli_probe(script, probe, ROOT)
    assert first["exit_code"] == 3
    assert first["machine_output_schema"]["properties"]["nested"]["properties"]["value"]["type"] == "integer"
    script.write_text('import json; print(json.dumps({"status": "ok"})); raise SystemExit(3)\n', encoding="utf-8")
    second = contract_module._single_cli_probe(script, probe, ROOT)
    left = _snapshot()
    right = copy.deepcopy(left)
    left["cli"]["synthetic_probe"] = {"probe": first}
    right["cli"]["synthetic_probe"] = {"probe": second}
    _rehash(left)
    _rehash(right)
    assert compare_snapshots(left, right)["classification"] == "breaking"

    script.write_text('import json; print(json.dumps({"status": "ok"})); raise SystemExit(2)\n', encoding="utf-8")
    with pytest.raises(ContractError, match="cli_probe_exit_code_drift"):
        contract_module._single_cli_probe(script, probe, ROOT)


def test_verify_current_fails_when_real_cli_fixture_is_replaced(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "src", tmp_path / "src")
    (tmp_path / "scripts").mkdir()
    cli_path = tmp_path / "scripts/check_run_provenance.py"
    shutil.copy2(ROOT / "scripts/check_run_provenance.py", cli_path)
    (tmp_path / "frontend/src/types").mkdir(parents=True)
    shutil.copy2(ROOT / "frontend/src/types/api.ts", tmp_path / "frontend/src/types/api.ts")
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["artifact_contracts"] = {}
    protocol["documentation_files"] = []
    protocol["cli_contracts"] = {
        "run_provenance": {
            "exit_codes": [0, 2, 3, 4],
            "path": "scripts/check_run_provenance.py",
            "probes": [
                {
                    "arguments": ["validate", "--manifest", "missing.json"],
                    "command": "validate",
                    "expected_exit_code": 2,
                    "probe_id": "validate",
                },
                {
                    "arguments": ["generate", "--spec", "missing.json", "--output", "out.json"],
                    "command": "generate",
                    "expected_exit_code": 4,
                    "probe_id": "generate",
                },
                {
                    "arguments": ["audit-legacy", "--profile", "missing.json"],
                    "command": "audit-legacy",
                    "expected_exit_code": 4,
                    "probe_id": "audit_legacy",
                },
            ],
        }
    }
    baseline = build_snapshot(protocol, repository_root=tmp_path)
    source = cli_path.read_text(encoding="utf-8")
    cli_path.write_text(
        source.replace(
            "print(json.dumps(report, ensure_ascii=False, sort_keys=True))",
            'report["synthetic_drift"] = True\n    print(json.dumps(report, ensure_ascii=False, sort_keys=True))',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="public_contract_drift"):
        verify_current(protocol, baseline, repository_root=tmp_path)


def test_recursive_artifact_schema_detects_nested_null_missing_and_type_drift() -> None:
    baseline = _snapshot()
    baseline["artifacts"]["synthetic"] = {
        "recursive_schema": recursive_schema({"outer": {"nullable": None, "count": 1}}),
        "strict_unknown_fields": True,
        "versions": {"schema_version": "1"},
    }
    _rehash(baseline)
    for payload in ({"outer": {"count": 1}}, {"outer": {"nullable": "", "count": 1}}):
        changed = copy.deepcopy(baseline)
        changed["artifacts"]["synthetic"]["recursive_schema"] = recursive_schema(payload)
        _rehash(changed)
        assert compare_snapshots(baseline, changed)["classification"] == "breaking"


def test_version_registry_requires_migration_and_supports_v1_to_v2_round_trip() -> None:
    baseline = _snapshot()
    with pytest.raises(ContractError, match="contract_migration_missing"):
        migrate_snapshot(baseline, target_version="2", migration_registry={})
    migrated = migrate_snapshot(
        baseline,
        target_version="2",
        migration_registry={"1->2": "v1_to_v2_envelope_v1"},
    )
    validate_snapshot(migrated, supported_versions={"2"})
    assert migrated["schema_version"] == "2"
    assert migrated["version_governance"]["supported_read_versions"] == ["1", "2"]
    tampered = copy.deepcopy(migrated)
    tampered["source_commit"] = "cross-version-tamper"
    with pytest.raises(ContractError, match="contract_snapshot_hash_mismatch"):
        validate_snapshot(tampered, supported_versions={"2"})


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ('{"schema_version":"1","schema_version":"2"}', "contract_json_duplicate_key"),
        ('{"value":NaN}', "contract_json_non_finite_number"),
    ],
)
def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers(
    tmp_path: Path, content: str, error: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ContractError, match=error):
        load_json(path)


def test_strict_json_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b"{\xff}")
    with pytest.raises(ContractError, match="contract_json_invalid"):
        load_json(path)


def test_cli_rejects_duplicate_keys_and_unmigrated_version_without_traceback(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"1","schema_version":"1"}\n', encoding="utf-8")
    duplicate_result = subprocess.run(
        [sys.executable, str(CLI), "verify-current", "--baseline", str(duplicate)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert duplicate_result.returncode == 2
    assert duplicate_result.stderr == ""

    future = copy.deepcopy(_snapshot())
    future["schema_version"] = "2"
    _rehash(future)
    future_path = tmp_path / "future.json"
    future_path.write_bytes(canonical_json(future))
    version_result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "compare",
            "--from",
            str(BASELINE_PATH),
            "--to",
            str(future_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert version_result.returncode == 2
    assert version_result.stderr == ""


def test_cli_reports_stable_exit_codes_without_traceback(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "verify-current", "--baseline", str(broken)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "breaking_or_versioning_violation"


def test_snapshot_round_trip_and_baseline_bytes_are_canonical(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    first = subprocess.run(
        [sys.executable, str(CLI), "snapshot", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert first.returncode == 0
    assert first.stdout == output.read_bytes()
    validate_snapshot(json.loads(first.stdout))
    second = subprocess.run(
        [sys.executable, str(CLI), "snapshot"], cwd=ROOT, capture_output=True, check=False
    )
    assert second.returncode == 0
    assert first.stdout == second.stdout


def test_snapshot_cannot_silently_replace_existing_baseline(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "snapshot", "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert output.read_text(encoding="utf-8") == "{}\n"
