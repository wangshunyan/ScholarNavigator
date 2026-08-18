from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.crash_consistency import stable_json_bytes
from scholar_agent.evaluation.provider_ingest_provenance import (
    BUNDLE_NAME,
    RAW_ARCHIVE_NAME,
    EncodingMetadata,
    ProviderAttemptEnvelope,
    ProviderCaptureRecorder,
    ProviderIngestError,
    ProviderIngestBundle,
    audit_frozen_record162,
    create_envelope,
    deterministic_fixture_matrix,
    parse_provider_bytes,
    verify_capture_bundle,
)
from scholar_agent.evaluation.resource_accounting import (
    ResourceLedgerV1,
    validate_resource_ledger,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_provider_ingest_provenance.py"


def _generate(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    report = deterministic_fixture_matrix(root)
    return (
        root / BUNDLE_NAME,
        root / RAW_ARCHIVE_NAME,
        root / "resource_ledger.json",
        report,
    )


def _rewrite_bundle(path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("bundle_sha256", None)
    value["bundle_sha256"] = stable_hash(value)
    path.write_bytes(stable_json_bytes(value))


def test_fixture_matrix_replays_all_sources_and_failures(tmp_path: Path) -> None:
    bundle, archive, ledger, report = _generate(tmp_path)
    assert report["exit_code"] == 0
    assert report["scenario_count"] == 17
    scenarios = {item["scenario"]: item for item in report["scenarios"]}
    assert scenarios["unknown_schema"]["rejected_record_count"] == 1
    assert scenarios["partial_page"] == {
        "scenario": "partial_page",
        "source": "openalex",
        "terminal_state": "partial_success",
        "accepted_record_count": 1,
        "rejected_record_count": 1,
    }
    assert scenarios["duplicate_records"]["accepted_record_count"] == 2
    assert verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)[
        "status"
    ] == "passed"
    resource = ResourceLedgerV1.model_validate_json(ledger.read_text(encoding="utf-8"))
    assert validate_resource_ledger(resource)["status"] == "passed"


def test_fixture_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = _generate(first)[3]
    two = _generate(second)[3]
    assert stable_json_bytes(one) == stable_json_bytes(two)
    for name in (BUNDLE_NAME, RAW_ARCHIVE_NAME, "resource_ledger.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_raw_hash_tamper_is_rejected(tmp_path: Path) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    report = verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)
    assert report["exit_code"] == 2
    assert report["violations"][0]["invariant"] == "raw_archive_hash_mismatch"


def test_count_nonconservation_fails_closed(tmp_path: Path) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        envelopes = value["envelopes"]
        assert isinstance(envelopes, list)
        envelopes[0]["accepted_record_count"] += 1

    _rewrite_bundle(bundle, mutate)
    report = verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)
    assert report["exit_code"] == 2
    assert report["violations"][0]["invariant"] == "bundle_or_archive_unreadable"


def test_pagination_cycle_and_duplicate_envelope_are_rejected(tmp_path: Path) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)

    def cycle(value: dict[str, object]) -> None:
        envelopes = value["envelopes"]
        assert isinstance(envelopes, list)
        candidates = [item for item in envelopes if item["parser_name"] == "openalex_search"]
        candidates[1]["request_cursor_sha256"] = candidates[0]["request_cursor_sha256"]

    _rewrite_bundle(bundle, cycle)
    report = verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)
    assert report["exit_code"] == 2
    assert any(item["invariant"] == "pagination_cursor_cycle" for item in report["violations"])

    bundle, archive, ledger, _ = _generate(tmp_path / "duplicate")

    def duplicate(value: dict[str, object]) -> None:
        envelopes = value["envelopes"]
        assert isinstance(envelopes, list)
        envelopes.append(copy.deepcopy(envelopes[-1]))

    _rewrite_bundle(bundle, duplicate)
    assert verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)[
        "exit_code"
    ] == 2


def test_ledger_attempt_and_generation_mismatch_are_rejected(tmp_path: Path) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)

    def mismatch(value: dict[str, object]) -> None:
        envelopes = value["envelopes"]
        assert isinstance(envelopes, list)
        envelopes[0]["checkpoint_generation"] = 2

    _rewrite_bundle(bundle, mismatch)
    report = verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)
    assert report["exit_code"] == 2
    assert report["violations"][0]["invariant"] == "bundle_or_archive_unreadable"


def test_replay_detects_output_hash_tamper(tmp_path: Path) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)

    def mutate(value: dict[str, object]) -> None:
        envelopes = value["envelopes"]
        assert isinstance(envelopes, list)
        envelopes[0]["parsed_output_sha256"] = "0" * 64

    _rewrite_bundle(bundle, mutate)
    report = verify_capture_bundle(bundle, archive, resource_ledger_path=ledger)
    assert report["exit_code"] == 2
    assert any(item["invariant"] == "parser_replay_mismatch" for item in report["violations"])


def test_encoding_malformed_json_and_gzip_are_explicit() -> None:
    encoding = EncodingMetadata(state="known", value="utf-8")
    malformed = parse_provider_bytes(
        "openalex",
        "openalex_search",
        b'{"results":[',
        encoding=encoding,
        compression="identity",
    )
    assert malformed.terminal_state == "malformed_response"
    invalid = parse_provider_bytes(
        "openalex",
        "openalex_search",
        b"\xff",
        encoding=encoding,
        compression="identity",
    )
    assert invalid.terminal_reason_code == "encoding_decode_failed"


def test_envelope_contains_no_request_url_headers_or_sensitive_values() -> None:
    run = hashlib.sha256(b"run").hexdigest()
    query = hashlib.sha256(b"query").hexdigest()
    attempt = hashlib.sha256(b"attempt").hexdigest()
    operation = hashlib.sha256(b"operation").hexdigest()
    manifest = hashlib.sha256(b"manifest").hexdigest()
    envelope, _ = create_envelope(
        run_identity=run,
        query_identity=query,
        source="openalex",
        attempt_identity=attempt,
        request_sequence=0,
        resource_operation_identity=operation,
        checkpoint_generation=0,
        manifest_identity=manifest,
        parser_name="openalex_search",
        raw_bytes=b'{"results":[]}',
        http_status=200,
        content_type="application/json; charset=utf-8",
        encoding=EncodingMetadata(state="known", value="utf-8"),
        compression="identity",
        terminal_state="success",
    )
    keys = set(envelope.model_dump(mode="json"))
    assert not {"url", "headers", "request_headers", "query_text"} & keys
    assert envelope.content_type == "application/json"


def test_capture_recorder_rejects_duplicate_request_sequence() -> None:
    values = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in ("run", "query", "attempt", "manifest", "operation")
    }
    recorder = ProviderCaptureRecorder(
        run_identity=values["run"],
        query_identity=values["query"],
        attempt_identity=values["attempt"],
        checkpoint_generation=1,
        manifest_identity=values["manifest"],
    )
    arguments = {
        "source": "openalex",
        "request_sequence": 0,
        "resource_operation_identity": values["operation"],
        "parser_name": "openalex_search",
        "raw_bytes": b'{"results":[]}',
        "http_status": 200,
        "content_type": "application/json",
        "encoding": EncodingMetadata(state="known", value="utf-8"),
        "compression": "identity",
        "terminal_state": "success",
    }
    recorder.record_attempt(**arguments)  # type: ignore[arg-type]
    with pytest.raises(ProviderIngestError, match="duplicate_request_sequence"):
        recorder.record_attempt(**arguments)  # type: ignore[arg-type]


def test_record162_is_not_eligible_and_cli_returns_three() -> None:
    report = audit_frozen_record162(ROOT)
    assert report["exit_code"] == 3
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "audit-frozen"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 3
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["raw_payload_inferred"] is False


def test_cli_fixture_double_run_is_identical() -> None:
    command = [sys.executable, str(SCRIPT), "check-fixtures"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout


@pytest.mark.parametrize("missing", ["bundle", "archive"])
def test_cli_missing_input_fails_without_traceback(tmp_path: Path, missing: str) -> None:
    bundle, archive, ledger, _ = _generate(tmp_path)
    if missing == "bundle":
        bundle = tmp_path / "missing.json"
    else:
        archive = tmp_path / "missing.tar"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--bundle",
            str(bundle),
            "--raw-archive",
            str(archive),
            "--resource-ledger",
            str(ledger),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == ""
    assert "Traceback" not in completed.stdout


def test_bundle_models_reject_extra_fields(tmp_path: Path) -> None:
    bundle_path, _archive, _ledger, _ = _generate(tmp_path)
    bundle = ProviderIngestBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    envelope = bundle.envelopes[0].model_dump(mode="json")
    envelope["request_url"] = "forbidden"
    with pytest.raises(Exception):
        ProviderAttemptEnvelope.model_validate(envelope)
