from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.provider_capacity_declaration_intake import (
    EVIDENCE_TYPES,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_VIOLATION,
    SOURCE_COMMIT,
    SOURCE_SCOPES,
    SOURCES,
    CapacityIntakeError,
    CapacityIntakeNotReady,
    _empty_ledger,
    _load_ledger,
    _synthetic_declaration,
    audit_readiness,
    build_declaration_contract,
    build_kit,
    build_launch_addendum,
    canonical_json,
    declaration_from_kit,
    declaration_template,
    import_declarations,
    load_protocol,
    read_kit,
    seal_declaration,
    simulate_matrix,
    stable_hash,
    validate_declaration,
    verify_import_receipt_for_launch,
    verify_kit,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "benchmark/provider_capacity_declaration_intake_v1_protocol.json"
)
CLI = ROOT / "scripts/check_provider_capacity_intake.py"
RUNTIME = ROOT / "scripts/provider_capacity_declaration_runtime.py"
EPOCH = 1_700_000_100


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _challenge(source: str) -> str:
    return hashlib.sha256(f"intake-test:{source}".encode()).hexdigest()


def _contract(protocol: dict[str, object], source: str) -> dict[str, object]:
    return build_declaration_contract(
        protocol,
        source=source,
        challenge_id=_challenge(source),
        issued_epoch=1_700_000_000,
    )


def _real_declaration(
    protocol: dict[str, object],
    source: str,
    *,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    contract = _contract(protocol, source)
    value = declaration_template(contract)
    value.update(
        {
            "declaration_version": f"{source}-capacity-v1",
            "limits": {
                "requests_per_second": 4,
                "requests_per_minute": 120,
                "burst": 8,
                "max_concurrency": 3,
                "cooldown_seconds": 5,
            },
            "valid_from_epoch": EPOCH - 1,
            "valid_until_epoch": EPOCH + 1000,
            "evidence_type": EVIDENCE_TYPES[0],
            "lifecycle_status": "active",
            "synthetic_only": False,
        }
    )
    if overrides:
        for key, replacement in overrides.items():
            if key.startswith("limits."):
                value["limits"][key.split(".", 1)[1]] = replacement
            else:
                value[key] = replacement
    return seal_declaration(value)


def _run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    for key in ("SystemRoot", "WINDIR"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _prepare_bundle(
    tmp_path: Path, protocol: dict[str, object]
) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    entries: dict[str, tuple[Path, Path]] = {}
    for source in SOURCES:
        kit = bundle / f"{source}.zip"
        declaration = bundle / f"{source}.declaration.json"
        build_kit(
            ROOT,
            protocol,
            source=source,
            challenge_id=_challenge(source),
            issued_epoch=1_700_000_000,
            output=kit,
        )
        write_json(declaration, _real_declaration(protocol, source))
        entries[source] = (kit, declaration)
    return bundle, entries


def test_protocol_binds_pacing_request_manifest_and_four_scopes(
    protocol: dict[str, object],
) -> None:
    assert protocol["source_commit"] == SOURCE_COMMIT
    assert protocol["population"] == {
        "http_attempt_upper": 19280,
        "logical_source_request_count": 9640,
        "query_count": 1000,
        "shard_count": 20,
        "sources": list(SOURCES),
    }
    assert protocol["source_scopes"] == SOURCE_SCOPES
    assert protocol["intake_policy"]["unknown_policy"] == (
        "not_available_fail_closed"
    )
    assert protocol["bindings"]["pacing_protocol"]["path"] == (
        "benchmark/formal_provider_pacing_v1_protocol.json"
    )


def test_contract_and_template_contain_no_endpoint_or_credential(
    protocol: dict[str, object],
) -> None:
    contract = _contract(protocol, "openalex")
    template = declaration_template(contract)
    encoded = canonical_json({"contract": contract, "template": template})
    lowered = encoded.lower()
    for forbidden in (
        b"api_key=",
        b"authorization",
        b"https://",
        b".env",
        str(ROOT).encode(),
    ):
        assert forbidden not in lowered
    assert contract["privacy"]["query_text_allowed"] is False
    assert template["limits"]["requests_per_second"] == "not_available"
    assert contract["privacy"]["free_text_allowed"] is False


def test_kit_is_deterministic_and_standard_library_only(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = build_kit(
        ROOT,
        protocol,
        source="arxiv",
        challenge_id=_challenge("arxiv"),
        issued_epoch=1_700_000_000,
        output=first,
    )
    two = build_kit(
        ROOT,
        protocol,
        source="arxiv",
        challenge_id=_challenge("arxiv"),
        issued_epoch=1_700_000_000,
        output=second,
    )
    assert first.read_bytes() == second.read_bytes()
    assert one == two
    assert verify_kit(first, protocol, repository_root=ROOT)["exit_code"] == 0
    manifest, files = read_kit(first)
    assert manifest["source"] == "arxiv"
    assert files["verify.py"] == RUNTIME.read_bytes()
    runtime = files["verify.py"].decode()
    assert "scholar_agent" not in runtime
    assert "subprocess" not in runtime


def test_two_no_repository_environments_verify_same_declaration(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    kit = tmp_path / "kit.zip"
    build_kit(
        ROOT,
        protocol,
        source="pubmed",
        challenge_id=_challenge("pubmed"),
        issued_epoch=1_700_000_000,
        output=kit,
    )
    declaration = _real_declaration(protocol, "pubmed")
    outputs: list[bytes] = []
    for name in ("first-site", "second-site"):
        site = tmp_path / name
        site.mkdir()
        with zipfile.ZipFile(kit) as archive:
            archive.extract("verify.py", site)
            archive.extract("declaration_contract.json", site)
        write_json(site / "declaration.json", declaration)
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(site / "home"),
            "TMPDIR": str(site / "tmp"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                str(site / "verify.py"),
                "verify",
                "--contract",
                str(site / "declaration_contract.json"),
                "--declaration",
                str(site / "declaration.json"),
                "--current-epoch",
                str(EPOCH),
            ],
            cwd=site,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == EXIT_READY
        assert result.stderr == b""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"limits.requests_per_second": -1}, "capacity_value_invalid"),
        ({"limits.max_concurrency": 0}, "capacity_value_invalid"),
        (
            {"limits.burst": 1, "limits.max_concurrency": 3},
            "capacity_burst_below_concurrency",
        ),
        (
            {"limits.requests_per_minute": 2, "limits.requests_per_second": 4},
            "capacity_window_contradiction",
        ),
        ({"api_scope_alias": "other_scope"}, "declaration_binding_invalid"),
        ({"lifecycle_status": "revoked"}, "declaration_revoked"),
        ({"evidence_type": "free text"}, "evidence_type_invalid"),
    ],
)
def test_invalid_structured_declarations_fail_closed(
    protocol: dict[str, object],
    overrides: dict[str, object],
    reason: str,
) -> None:
    contract = _contract(protocol, "semantic_scholar")
    value = _real_declaration(
        protocol, "semantic_scholar", overrides=overrides
    )
    with pytest.raises(CapacityIntakeError, match=reason):
        validate_declaration(
            contract, value, current_epoch=EPOCH, allow_synthetic=False
        )


def test_expired_and_tampered_declarations_fail_closed(
    protocol: dict[str, object],
) -> None:
    contract = _contract(protocol, "arxiv")
    expired = _real_declaration(
        protocol,
        "arxiv",
        overrides={
            "valid_from_epoch": EPOCH - 100,
            "valid_until_epoch": EPOCH - 1,
        },
    )
    with pytest.raises(CapacityIntakeError, match="expired"):
        validate_declaration(contract, expired, current_epoch=EPOCH)
    tampered = _real_declaration(protocol, "arxiv")
    tampered["limits"]["burst"] += 1
    with pytest.raises(CapacityIntakeError, match="digest"):
        validate_declaration(contract, tampered, current_epoch=EPOCH)


def test_import_is_single_use_and_preserves_9640_request_set(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    _bundle, entries = _prepare_bundle(tmp_path, protocol)
    ledger = tmp_path / "ledger.json"
    receipt = import_declarations(
        ROOT,
        protocol,
        entries=entries,
        ledger_path=ledger,
        current_epoch=EPOCH,
    )
    preservation = receipt["request_preservation"]
    assert preservation["intent_count"] == 9640
    assert preservation["http_attempt_upper"] == 19280
    assert preservation["shard_count"] == 20
    assert preservation["request_set_unchanged"] is True
    assert preservation["request_parameter_mutation_count"] == 0
    assert preservation["duplicate_request_count"] == 0
    assert preservation["window_violation_count"] == 0
    assert receipt["launch_activation_allowed"] is True
    verify_import_receipt_for_launch(receipt, protocol)
    drifted = copy.deepcopy(receipt)
    drifted["launch_control_sha256"] = "0" * 64
    payload = dict(drifted)
    payload.pop("receipt_sha256")
    drifted["receipt_sha256"] = stable_hash(payload)
    with pytest.raises(CapacityIntakeError, match="launch_binding"):
        verify_import_receipt_for_launch(drifted, protocol)
    with pytest.raises(CapacityIntakeError, match="challenge_replay"):
        import_declarations(
            ROOT,
            protocol,
            entries=entries,
            ledger_path=ledger,
            current_epoch=EPOCH,
        )


def test_missing_source_and_cross_source_kit_fail(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    _bundle, entries = _prepare_bundle(tmp_path, protocol)
    incomplete = dict(entries)
    incomplete.pop("pubmed")
    with pytest.raises(CapacityIntakeNotReady, match="missing"):
        import_declarations(
            ROOT,
            protocol,
            entries=incomplete,
            ledger_path=tmp_path / "missing-ledger.json",
            current_epoch=EPOCH,
        )
    crossed = dict(entries)
    crossed["openalex"] = entries["arxiv"]
    with pytest.raises(CapacityIntakeError, match="source_kit_mismatch"):
        import_declarations(
            ROOT,
            protocol,
            entries=crossed,
            ledger_path=tmp_path / "cross-ledger.json",
            current_epoch=EPOCH,
        )


def test_challenge_ledger_detects_tamper_reorder_and_duplicate(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    _bundle, entries = _prepare_bundle(tmp_path, protocol)
    ledger = tmp_path / "ledger.json"
    import_declarations(
        ROOT,
        protocol,
        entries=entries,
        ledger_path=ledger,
        current_epoch=EPOCH,
    )
    value = json.loads(ledger.read_text())
    assert len(value["events"]) == 4
    tampered = copy.deepcopy(value)
    tampered["events"][0]["source"] = "arxiv"
    tampered["ledger_sha256"] = stable_hash(tampered["events"])
    write_json(tmp_path / "tampered.json", tampered)
    with pytest.raises(CapacityIntakeError, match="chain"):
        _load_ledger(tmp_path / "tampered.json")
    duplicate = copy.deepcopy(value)
    duplicate["events"].append(copy.deepcopy(duplicate["events"][-1]))
    duplicate["ledger_sha256"] = stable_hash(duplicate["events"])
    write_json(tmp_path / "duplicate.json", duplicate)
    with pytest.raises(CapacityIntakeError, match="chain"):
        _load_ledger(tmp_path / "duplicate.json")


def test_matrix_covers_all_pre_registered_scenarios_deterministically(
    protocol: dict[str, object],
) -> None:
    first = simulate_matrix(ROOT, protocol)
    second = simulate_matrix(ROOT, protocol)
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == 11
    assert first["accepted_scenario_count"] == 2
    rows = {row["scenario"]: row for row in first["scenarios"]}
    assert rows["qualified_four_sources"]["request_preservation"][
        "intent_count"
    ] == 9640
    assert rows["dynamic_reduction"]["request_preservation"][
        "request_set_unchanged"
    ] is True
    assert rows["single_source_missing"]["status"] == "not_ready"
    assert rows["challenge_replay"]["reason_code"] == "challenge_replay"
    assert rows["revoked"]["reason_code"] == "declaration_revoked"


def test_real_readiness_lists_all_four_sources_and_launch_stays_blocked(
    protocol: dict[str, object],
) -> None:
    report = audit_readiness(protocol)
    assert report["exit_code"] == EXIT_NOT_READY
    assert report["activation_allowed"] is False
    assert [row["source"] for row in report["missing_declarations"]] == list(
        SOURCES
    )
    addendum = build_launch_addendum(protocol, report)
    assert addendum["request_set_mutated"] is False
    assert addendum["logical_source_request_count"] == 9640
    assert addendum["http_attempt_upper"] == 19280
    assert addendum["formal_validation_complete"] is False


def test_kit_tamper_extra_member_and_duplicate_member_fail(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    original = tmp_path / "original.zip"
    build_kit(
        ROOT,
        protocol,
        source="openalex",
        challenge_id=_challenge("openalex"),
        issued_epoch=1_700_000_000,
        output=original,
    )
    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(extra, "w") as out:
        for info in source.infolist():
            out.writestr(info, source.read(info))
        out.writestr("extra.txt", b"unexpected")
    with pytest.raises(CapacityIntakeError, match="inventory"):
        verify_kit(extra, protocol, repository_root=ROOT)
    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(
            duplicate, "w"
        ) as out:
            for info in source.infolist():
                out.writestr(info, source.read(info))
            out.writestr("README.txt", b"second")
    with pytest.raises(CapacityIntakeError, match="unsafe"):
        read_kit(duplicate)


def test_cli_matrix_readiness_and_usage_are_stable_no_traceback(
    tmp_path: Path,
) -> None:
    first = _run_cli("simulate-matrix")
    second = _run_cli("simulate-matrix")
    assert first.returncode == second.returncode == EXIT_READY
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    readiness = _run_cli("audit-readiness")
    assert readiness.returncode == EXIT_NOT_READY
    assert json.loads(readiness.stdout)["missing_declaration_count"] == 4
    assert readiness.stderr == b""
    usage = _run_cli()
    assert usage.returncode == 4
    assert usage.stderr == b""
    assert b"Traceback" not in usage.stdout
    missing = _run_cli(
        "verify-declaration",
        "--kit",
        str(tmp_path / "missing.zip"),
        "--declaration",
        str(tmp_path / "missing.json"),
        "--current-epoch",
        str(EPOCH),
    )
    assert missing.returncode == EXIT_VIOLATION
    assert missing.stderr == b""
    assert b"Traceback" not in missing.stdout


def test_cli_build_verify_and_import_dry_run(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    bundle, _entries = _prepare_bundle(tmp_path, protocol)
    verify = _run_cli(
        "verify-declaration",
        "--kit",
        str(bundle / "openalex.zip"),
        "--declaration",
        str(bundle / "openalex.declaration.json"),
        "--current-epoch",
        str(EPOCH),
    )
    assert verify.returncode == EXIT_READY
    assert verify.stderr == b""
    ingest = _run_cli(
        "import-dry-run",
        "--bundle-dir",
        str(bundle),
        "--ledger",
        str(tmp_path / "cli-ledger.json"),
        "--current-epoch",
        str(EPOCH),
    )
    assert ingest.returncode == EXIT_READY
    report = json.loads(ingest.stdout)
    assert report["request_preservation"]["intent_count"] == 9640
    assert report["request_preservation"]["request_set_unchanged"] is True


def test_protocol_cross_commit_and_scope_tamper_fail(
    tmp_path: Path, protocol: dict[str, object]
) -> None:
    for key, replacement in (
        ("source_commit", "0" * 40),
        ("source_scopes", {**SOURCE_SCOPES, "arxiv": "wrong"}),
    ):
        changed = copy.deepcopy(protocol)
        changed[key] = replacement
        payload = copy.deepcopy(changed)
        payload.pop("protocol_sha256")
        changed["protocol_sha256"] = stable_hash(payload)
        path = tmp_path / f"{key}.json"
        write_json(path, changed)
        with pytest.raises(CapacityIntakeError):
            load_protocol(path, repository_root=ROOT)


def test_readiness_freshness_and_public_contract_integrations() -> None:
    readiness = json.loads(
        (
            ROOT / "benchmark/validation_readiness_bundle_v1_contract.json"
        ).read_text()
    )
    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"]
        == "architecture_provider_capacity_declaration_intake_ready"
    )
    assert claim["status"] == "verified"
    assert claim["scope"] == "engineering_capability"
    evidence = {
        row["evidence_id"]: row for row in readiness["evidence"]
    }
    assert evidence["provider_capacity_intake_readiness"]["checks"] == [
        {
            "equals": "not_ready_missing_real_declarations",
            "pointer": "/status",
        },
        {"equals": 4, "pointer": "/missing_declaration_count"},
        {"equals": False, "pointer": "/activation_allowed"},
    ]
    gate = next(
        row
        for row in readiness["read_only_gates"]
        if row["gate_id"] == "provider_capacity_declaration_intake"
    )
    assert gate["expected_exit_code"] == EXIT_NOT_READY
    assert readiness["release"]["status"] == "ready_with_declared_blockers"
    assert len(readiness["blockers"]) == 3

    freshness = json.loads(
        (
            ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json"
        ).read_text()
    )
    assert (
        freshness["claim_component_bindings"][
            "architecture_provider_capacity_declaration_intake_ready"
        ]
        == ["provider_capacity_declaration_intake"]
    )
    assert "provider_capacity_intake_readiness" in freshness[
        "blocked_evidence_ids"
    ]

    public = json.loads(
        (
            ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
        ).read_text()
    )
    assert public["artifact_contracts"][
        "provider_capacity_declaration_intake"
    ].endswith("provider_capacity_declaration_intake_v1_protocol.json")
    contract = public["cli_contracts"][
        "provider_capacity_declaration_intake"
    ]
    assert contract["exit_codes"] == [0, 2, 3, 4]
    assert {row["command"] for row in contract["probes"]} == {
        "build-kit",
        "verify-declaration",
        "import-dry-run",
        "simulate-matrix",
        "audit-readiness",
    }
