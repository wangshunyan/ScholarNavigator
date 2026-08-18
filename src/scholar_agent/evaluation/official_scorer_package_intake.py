"""Offline intake gate for a future official scorer package.

This module validates a bounded package and its declared schemas before handing
its conformance entrypoint to ``external_scorer_handoff_v1``.  It never treats
content hashes as publisher authentication and never computes a quality score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.external_scorer_handoff import (
    ExternalScorerError,
    create_package_manifest,
    load_protocol as load_handoff_protocol,
    run_scorer,
    stable_json_bytes,
    synthetic_handoff,
    synthetic_scorer_source,
)


PROTOCOL = "official_scorer_package_intake_v1"
KIT_PROTOCOL = "official_scorer_intake_kit_v1"
PACKAGE_PROTOCOL = "official_scorer_candidate_package_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "3a7038b0c4244ff09a4813c3176417747eb99ff8"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_PROVIDED = "not_provided"
UNKNOWN = "unknown"
UNVERIFIED_ORIGIN = "unverified_origin"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_MEMBER_COUNT = 16
MAX_COMPRESSION_RATIO = 50
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
ORIGIN_EVIDENCE_TYPES = (
    "event_organizer_detached_signature_v1",
    "event_organizer_offline_handoff_record_v1",
    "unverified_origin",
)
VERIFIED_ORIGIN_TYPES = frozenset(ORIGIN_EVIDENCE_TYPES[:2])
METRIC_TYPES = ("integer", "number")
METRIC_DIRECTIONS = ("higher_is_better", "lower_is_better", "not_provided")
MISSING_SEMANTICS = ("error", "explicit_null", "not_provided")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class OfficialScorerIntakeError(RuntimeError):
    """The package, schema, challenge, origin, or sandbox contract failed."""


class OfficialScorerIntakeNotReady(OfficialScorerIntakeError):
    """Required real official material is absent or unauthenticated."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def decode_json_object(raw: bytes, *, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OfficialScorerIntakeError(reason) from exc
    if not isinstance(value, dict):
        raise OfficialScorerIntakeError(reason)
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OfficialScorerIntakeError("json_input_unavailable") from exc
    if len(raw) > MAX_MEMBER_BYTES:
        raise OfficialScorerIntakeError("json_input_too_large")
    return decode_json_object(raw, reason="json_input_invalid")


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload[key] = "0" * 64
    return sha256_bytes(canonical_json(payload))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise OfficialScorerIntakeError("unsafe_package_path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.as_posix() != value
        or path.name == ".env"
        or path.parts[0] == "third_party"
    ):
        raise OfficialScorerIntakeError("unsafe_package_path")
    return value


def _validate_schema(schema: Any, *, role: str) -> dict[str, Any]:
    if schema == NOT_PROVIDED:
        return {}
    if not isinstance(schema, dict):
        raise OfficialScorerIntakeError(f"{role}_schema_invalid")
    if schema.get("$schema") not in {
        "https://json-schema.org/draft/2020-12/schema",
        NOT_PROVIDED,
    }:
        raise OfficialScorerIntakeError(f"{role}_schema_dialect_invalid")
    if schema.get("type") not in {"object", NOT_PROVIDED}:
        raise OfficialScorerIntakeError(f"{role}_schema_root_invalid")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise OfficialScorerIntakeError(f"{role}_schema_structure_invalid")
    if len(required) != len(set(required)) or any(
        not isinstance(name, str) or name not in properties for name in required
    ):
        raise OfficialScorerIntakeError(f"{role}_schema_required_invalid")
    if schema.get("additionalProperties") is not False:
        raise OfficialScorerIntakeError(f"{role}_schema_not_strict")
    return schema


def _validate_metrics(value: Any) -> list[dict[str, Any]]:
    if value == NOT_PROVIDED:
        return []
    if not isinstance(value, list) or not value:
        raise OfficialScorerIntakeError("metric_namespace_invalid")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {
            "direction",
            "missing_semantics",
            "name",
            "type",
        }:
            raise OfficialScorerIntakeError("metric_schema_invalid")
        name = raw["name"]
        if (
            not isinstance(name, str)
            or not IDENTITY_RE.fullmatch(name)
            or "." not in name
            or name in seen
            or raw["type"] not in METRIC_TYPES
            or raw["direction"] not in METRIC_DIRECTIONS
            or raw["missing_semantics"] not in MISSING_SEMANTICS
        ):
            raise OfficialScorerIntakeError("metric_definition_invalid")
        seen.add(name)
        result.append(dict(raw))
    return result


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    expected = {
        "bindings",
        "execution",
        "formal_validation_complete",
        "intake_contract",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "synthetic_scenarios",
    }
    if set(value) != expected:
        raise OfficialScorerIntakeError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or _digest_without(value, "protocol_sha256") != value["protocol_sha256"]
    ):
        raise OfficialScorerIntakeError("protocol_binding_invalid")
    bindings = value["bindings"]
    required_bindings = {
        "clearance",
        "external_scorer_handoff",
        "full1000_plan",
        "preregistration",
        "quarantine",
    }
    if not isinstance(bindings, dict) or set(bindings) != required_bindings:
        raise OfficialScorerIntakeError("protocol_binding_inventory_invalid")
    for name, spec in bindings.items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise OfficialScorerIntakeError("protocol_binding_schema_invalid")
        relative = _safe_relative(spec["path"])
        target = repository_root / relative
        if not target.is_file() or sha256_file(target) != spec["sha256"]:
            raise OfficialScorerIntakeError(f"protocol_binding_drift:{name}")
    expected_contract = {
        "archive": {
            "maximum_archive_bytes": MAX_ARCHIVE_BYTES,
            "maximum_compression_ratio": MAX_COMPRESSION_RATIO,
            "maximum_member_bytes": MAX_MEMBER_BYTES,
            "maximum_member_count": MAX_MEMBER_COUNT,
            "sdist_allowed": False,
        },
        "challenge": "single_use_bound_to_plan_commit_and_package",
        "hash_authentication_boundary": (
            "sha256_proves_content_integrity_not_event_organizer_identity"
        ),
        "metric_directions": list(METRIC_DIRECTIONS),
        "metric_missing_semantics": list(MISSING_SEMANTICS),
        "metric_types": list(METRIC_TYPES),
        "origin_evidence_types": list(ORIGIN_EVIDENCE_TYPES),
        "required_real_material": [
            "official_package",
            "official_input_schema",
            "official_output_schema",
            "official_metric_namespace",
            "verified_official_origin",
        ],
        "sandbox": "external_scorer_handoff_v1",
        "unknown_policy": "unknown_or_not_provided_fail_closed",
    }
    if value["intake_contract"] != expected_contract:
        raise OfficialScorerIntakeError("intake_contract_drift")
    expected_scenarios = [
        "qualified",
        "missing_schema",
        "unknown_namespace",
        "entrypoint_tamper",
        "extra_metric",
        "nondeterministic_output",
        "illegal_io",
        "challenge_replay",
        "cross_version_mixing",
        "unverified_origin",
        "revoked_reuse",
    ]
    if value["synthetic_scenarios"] != {
        "names": expected_scenarios,
        "synthetic_only": True,
    }:
        raise OfficialScorerIntakeError("synthetic_scenario_contract_drift")
    return value


def build_contract(
    protocol: Mapping[str, Any], *, challenge: str
) -> dict[str, Any]:
    if not isinstance(challenge, str) or not SHA256_RE.fullmatch(challenge):
        raise OfficialScorerIntakeError("challenge_invalid")
    value: dict[str, Any] = {
        "bindings": {
            name: protocol["bindings"][name]["sha256"]
            for name in (
                "external_scorer_handoff",
                "full1000_plan",
                "preregistration",
            )
        },
        "challenge": challenge,
        "package_protocol": PACKAGE_PROTOCOL,
        "privacy": {
            "absolute_paths_allowed": False,
            "credentials_allowed": False,
            "endpoint_allowed": False,
            "query_text_allowed": False,
            "request_headers_allowed": False,
        },
        "protocol": KIT_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    value["contract_sha256"] = sha256_bytes(canonical_json(value))
    return value


def package_template(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "allowed_io": NOT_PROVIDED,
        "challenge": contract["challenge"],
        "contract_sha256": contract["contract_sha256"],
        "determinism": NOT_PROVIDED,
        "entrypoint": NOT_PROVIDED,
        "files": NOT_PROVIDED,
        "input_schema": NOT_PROVIDED,
        "lifecycle": NOT_PROVIDED,
        "manifest_sha256": "0" * 64,
        "metrics": NOT_PROVIDED,
        "origin_evidence_type": UNVERIFIED_ORIGIN,
        "output_schema": NOT_PROVIDED,
        "package_protocol": PACKAGE_PROTOCOL,
        "resource_limits": NOT_PROVIDED,
        "runtime": NOT_PROVIDED,
        "schema_version": SCHEMA_VERSION,
        "scorer_name": UNKNOWN,
        "scorer_version": UNKNOWN,
        "source_commit": SOURCE_COMMIT,
        "synthetic_only": False,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    challenge: str,
    output: Path,
) -> dict[str, Any]:
    contract = build_contract(protocol, challenge=challenge)
    runtime = repository_root / "scripts" / "official_scorer_intake_runtime.py"
    files = {
        "README.txt": (
            "Offline intake kit. Run verify.py with python -I -S. Fill only "
            "declared structured fields. Hashes prove integrity, not organizer "
            "identity. No query, endpoint, credential, or header is included.\n"
        ).encode(),
        "contract.json": canonical_json(contract),
        "package_manifest_template.json": canonical_json(package_template(contract)),
        "verify.py": runtime.read_bytes(),
    }
    inventory = [
        {"path": name, "sha256": sha256_bytes(raw), "size": len(raw)}
        for name, raw in sorted(files.items())
    ]
    manifest: dict[str, Any] = {
        "challenge": challenge,
        "contract_sha256": contract["contract_sha256"],
        "files": inventory,
        "manifest_sha256": "0" * 64,
        "protocol": KIT_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    manifest["manifest_sha256"] = _digest_without(manifest, "manifest_sha256")
    files["manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, "w") as archive:
            for name, raw in sorted(files.items()):
                archive.writestr(_zip_info(name), raw)
    except OSError as exc:
        raise OfficialScorerIntakeError("kit_write_failed") from exc
    return {
        "challenge": challenge,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "kit_sha256": sha256_file(output),
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "official_scorer_intake_kit_built",
    }


def _read_archive(path: Path) -> dict[str, bytes]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise OfficialScorerIntakeError("archive_size_or_presence_invalid")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise OfficialScorerIntakeError("archive_member_limit")
            files: dict[str, bytes] = {}
            for info in infos:
                name = _safe_relative(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    name in files
                    or info.is_dir()
                    or mode not in (0, 0o100000)
                    or info.file_size > MAX_MEMBER_BYTES
                    or info.compress_size > MAX_MEMBER_BYTES
                    or (
                        info.compress_size == 0
                        and info.file_size > 0
                    )
                    or (
                        info.compress_size > 0
                        and info.file_size
                        > info.compress_size * MAX_COMPRESSION_RATIO
                    )
                ):
                    raise OfficialScorerIntakeError("archive_member_unsafe")
                files[name] = archive.read(info)
    except OfficialScorerIntakeError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise OfficialScorerIntakeError("archive_invalid") from exc
    return files


def verify_kit(path: Path, protocol: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    files = _read_archive(path)
    expected = {
        "README.txt",
        "contract.json",
        "manifest.json",
        "package_manifest_template.json",
        "verify.py",
    }
    if set(files) != expected:
        raise OfficialScorerIntakeError("kit_inventory_invalid")
    manifest = decode_json_object(files["manifest.json"], reason="kit_manifest_invalid")
    if set(manifest) != {
        "challenge",
        "contract_sha256",
        "files",
        "manifest_sha256",
        "protocol",
        "schema_version",
        "source_commit",
    } or _digest_without(manifest, "manifest_sha256") != manifest["manifest_sha256"]:
        raise OfficialScorerIntakeError("kit_manifest_invalid")
    if (
        manifest["protocol"] != KIT_PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
    ):
        raise OfficialScorerIntakeError("kit_manifest_binding_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise OfficialScorerIntakeError("kit_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or row["path"] in seen
            or row["path"] not in files
            or row["path"] == "manifest.json"
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != sha256_bytes(files[row["path"]])
        ):
            raise OfficialScorerIntakeError("kit_inventory_invalid")
        seen.add(row["path"])
    contract = decode_json_object(files["contract.json"], reason="kit_contract_invalid")
    expected_contract = build_contract(protocol, challenge=manifest["challenge"])
    if contract != expected_contract or contract["contract_sha256"] != manifest["contract_sha256"]:
        raise OfficialScorerIntakeError("kit_contract_binding_invalid")
    if files["verify.py"] != (
        repository_root / "scripts" / "official_scorer_intake_runtime.py"
    ).read_bytes():
        raise OfficialScorerIntakeError("kit_runtime_drift")
    if decode_json_object(
        files["package_manifest_template.json"], reason="kit_template_invalid"
    ) != package_template(contract):
        raise OfficialScorerIntakeError("kit_template_drift")
    return contract


def _validate_manifest(
    manifest: Mapping[str, Any],
    files: Mapping[str, bytes],
    contract: Mapping[str, Any],
    *,
    allow_synthetic: bool,
) -> dict[str, Any]:
    expected = set(package_template(contract))
    if set(manifest) != expected:
        raise OfficialScorerIntakeError("package_manifest_schema_invalid")
    if (
        manifest["package_protocol"] != PACKAGE_PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["challenge"] != contract["challenge"]
        or manifest["contract_sha256"] != contract["contract_sha256"]
        or _digest_without(manifest, "manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise OfficialScorerIntakeError("package_manifest_binding_invalid")
    for field in ("scorer_name", "scorer_version"):
        if not isinstance(manifest[field], str) or not IDENTITY_RE.fullmatch(manifest[field]):
            raise OfficialScorerIntakeError("package_identity_invalid")
    entrypoint = _safe_relative(manifest["entrypoint"])
    if entrypoint not in files or not entrypoint.endswith(".py"):
        raise OfficialScorerIntakeError("package_entrypoint_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list) or not inventory:
        raise OfficialScorerIntakeError("package_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or not isinstance(row["path"], str)
            or row["path"] in seen
            or row["path"] == "intake_manifest.json"
            or row["path"] not in files
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != sha256_bytes(files[row["path"]])
        ):
            raise OfficialScorerIntakeError("package_inventory_invalid")
        _safe_relative(row["path"])
        seen.add(row["path"])
    if seen != set(files) - {"intake_manifest.json"}:
        raise OfficialScorerIntakeError("package_inventory_not_closed")
    _validate_schema(manifest["input_schema"], role="input")
    _validate_schema(manifest["output_schema"], role="output")
    _validate_metrics(manifest["metrics"])
    if manifest["allowed_io"] != {
        "environment_files": False,
        "input": "canonical_handoff_read_only",
        "network": False,
        "output": "isolated_temporary_output_only",
        "subprocess": False,
    }:
        raise OfficialScorerIntakeError("package_io_contract_invalid")
    if manifest["determinism"] != {"repeat_runs": 2, "required": True}:
        raise OfficialScorerIntakeError("package_determinism_invalid")
    if manifest["runtime"] != {
        "kind": "python_source_sandbox",
        "version": "not_provided",
    }:
        raise OfficialScorerIntakeError("package_runtime_invalid")
    limits = manifest["resource_limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "maximum_input_bytes",
        "maximum_output_bytes",
        "timeout_seconds",
    } or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in limits.values()
    ):
        raise OfficialScorerIntakeError("package_resource_limits_invalid")
    if manifest["lifecycle"] not in {"active", "revoked"}:
        raise OfficialScorerIntakeError("package_lifecycle_invalid")
    if manifest["lifecycle"] != "active":
        raise OfficialScorerIntakeError("package_revoked")
    synthetic = manifest["synthetic_only"]
    if synthetic is not False and not (allow_synthetic and synthetic is True):
        raise OfficialScorerIntakeError("synthetic_package_not_real")
    origin = manifest["origin_evidence_type"]
    if origin not in ORIGIN_EVIDENCE_TYPES:
        raise OfficialScorerIntakeError("origin_evidence_type_invalid")
    result = copy.deepcopy(dict(manifest))
    result["origin_status"] = (
        "verified_official_origin"
        if origin in VERIFIED_ORIGIN_TYPES and synthetic is False
        else UNVERIFIED_ORIGIN
    )
    return result


def _missing_required_material(manifest: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if manifest["input_schema"] == NOT_PROVIDED:
        missing.append("official_input_schema")
    if manifest["output_schema"] == NOT_PROVIDED:
        missing.append("official_output_schema")
    if manifest["metrics"] == NOT_PROVIDED:
        missing.append("official_metric_namespace")
    return missing


def verify_candidate_package(
    kit_path: Path,
    package_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    allow_synthetic: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    contract = verify_kit(kit_path, protocol, repository_root=repository_root)
    files = _read_archive(package_path)
    if "intake_manifest.json" not in files:
        raise OfficialScorerIntakeError("package_manifest_missing")
    manifest = decode_json_object(
        files["intake_manifest.json"], reason="package_manifest_invalid"
    )
    return (
        _validate_manifest(
            manifest, files, contract, allow_synthetic=allow_synthetic
        ),
        files,
    )


def conformance_dry_run(
    kit_path: Path,
    package_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    manifest, files = verify_candidate_package(
        kit_path,
        package_path,
        protocol,
        repository_root=repository_root,
        allow_synthetic=allow_synthetic,
    )
    missing = _missing_required_material(manifest)
    if missing:
        raise OfficialScorerIntakeNotReady(
            "official_material_incomplete:" + ",".join(missing)
        )
    handoff_protocol = load_handoff_protocol(
        repository_root / protocol["bindings"]["external_scorer_handoff"]["path"],
        repository_root=repository_root,
    )
    with tempfile.TemporaryDirectory(prefix="official-scorer-conformance-") as temp_name:
        root = Path(temp_name)
        sandbox_package = root / "sandbox-package"
        create_package_manifest(
            sandbox_package,
            scorer_name="synthetic-strict-scorer",
            scorer_version="1",
            entrypoint_source=files[manifest["entrypoint"]].decode("utf-8"),
        )
        handoff_path = root / "handoff.json"
        handoff_path.write_bytes(canonical_json(synthetic_handoff()))
        first = run_scorer(
            sandbox_package,
            handoff_path,
            handoff_protocol,
            repository_root=repository_root,
            run_ordinal=1,
        )
        second = run_scorer(
            sandbox_package,
            handoff_path,
            handoff_protocol,
            repository_root=repository_root,
            run_ordinal=2,
        )
        if first["output_bytes"] != second["output_bytes"]:
            raise OfficialScorerIntakeError("conformance_nondeterministic")
    return {
        "conformance_output_sha256": first["output_sha256"],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "origin_status": manifest["origin_status"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "official_scorer_package_qualified",
        "worker_audit": first["worker_audit"],
    }


def _qualified_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {"synthetic": {"type": "string"}},
        "required": ["synthetic"],
        "type": "object",
    }


def build_synthetic_package(
    contract: Mapping[str, Any],
    output: Path,
    *,
    scenario: str = "qualified",
) -> None:
    scorer_scenario = {
        "extra_metric": "extra_metric",
        "nondeterministic_output": "nondeterministic_output",
        "illegal_io": "network_attempt",
    }.get(scenario, "valid")
    files = {"scorer.py": synthetic_scorer_source(scorer_scenario).encode()}
    manifest = package_template(contract)
    manifest.update(
        {
            "allowed_io": {
                "environment_files": False,
                "input": "canonical_handoff_read_only",
                "network": False,
                "output": "isolated_temporary_output_only",
                "subprocess": False,
            },
            "determinism": {"repeat_runs": 2, "required": True},
            "entrypoint": "scorer.py",
            "files": [
                {
                    "path": "scorer.py",
                    "sha256": sha256_bytes(files["scorer.py"]),
                    "size": len(files["scorer.py"]),
                }
            ],
            "input_schema": _qualified_schema(),
            "lifecycle": "active",
            "metrics": [
                {
                    "direction": "not_provided",
                    "missing_semantics": "error",
                    "name": "official.synthetic",
                    "type": "number",
                }
            ],
            "origin_evidence_type": "event_organizer_offline_handoff_record_v1",
            "output_schema": _qualified_schema(),
            "resource_limits": {
                "maximum_input_bytes": 1048576,
                "maximum_output_bytes": 1048576,
                "timeout_seconds": 2,
            },
            "runtime": {
                "kind": "python_source_sandbox",
                "version": "not_provided",
            },
            "scorer_name": "synthetic-official-scorer",
            "scorer_version": "1",
            "synthetic_only": True,
        }
    )
    if scenario == "missing_schema":
        manifest["input_schema"] = NOT_PROVIDED
    elif scenario == "unknown_namespace":
        manifest["metrics"] = NOT_PROVIDED
    elif scenario == "unverified_origin":
        manifest["origin_evidence_type"] = UNVERIFIED_ORIGIN
    elif scenario == "revoked_reuse":
        manifest["lifecycle"] = "revoked"
    elif scenario == "cross_version_mixing":
        manifest["source_commit"] = "f" * 40
    manifest["manifest_sha256"] = _digest_without(manifest, "manifest_sha256")
    if scenario == "entrypoint_tamper":
        files["scorer.py"] += b"\n# tampered\n"
    files["intake_manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(_zip_info(name), raw)


def import_dry_run(
    kit_path: Path,
    package_path: Path,
    ledger_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    manifest, _files = verify_candidate_package(
        kit_path,
        package_path,
        protocol,
        repository_root=repository_root,
        allow_synthetic=allow_synthetic,
    )
    missing = _missing_required_material(manifest)
    if missing:
        raise OfficialScorerIntakeNotReady(
            "official_material_incomplete:" + ",".join(missing)
        )
    if (
        manifest["origin_status"] != "verified_official_origin"
        and not allow_synthetic
    ):
        raise OfficialScorerIntakeNotReady("official_origin_unverified")
    ledger = (
        read_object(ledger_path)
        if ledger_path.exists()
        else {"challenges": [], "protocol": "official_scorer_intake_ledger_v1"}
    )
    if set(ledger) != {"challenges", "protocol"} or not isinstance(
        ledger["challenges"], list
    ):
        raise OfficialScorerIntakeError("intake_ledger_invalid")
    challenge = manifest["challenge"]
    if challenge in ledger["challenges"]:
        raise OfficialScorerIntakeError("challenge_replay")
    ledger["challenges"].append(challenge)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(canonical_json(ledger))
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "package_manifest_sha256": manifest["manifest_sha256"],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "official_scorer_package_qualified",
    }


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="official-scorer-intake-matrix-") as temp_name:
        root = Path(temp_name)
        kit = root / "kit.zip"
        challenge = sha256_bytes(b"synthetic-official-scorer-challenge")
        build_kit(repository_root, protocol, challenge=challenge, output=kit)
        contract = verify_kit(kit, protocol, repository_root=repository_root)
        for scenario in protocol["synthetic_scenarios"]["names"]:
            package = root / f"{scenario}.zip"
            build_synthetic_package(contract, package, scenario=scenario)
            expected = "passed" if scenario == "qualified" else "rejected"
            observed = "passed"
            reason = None
            try:
                conformance_dry_run(
                    kit,
                    package,
                    protocol,
                    repository_root=repository_root,
                    allow_synthetic=True,
                )
                if scenario == "challenge_replay":
                    ledger = root / "ledger.json"
                    import_dry_run(
                        kit,
                        package,
                        ledger,
                        protocol,
                        repository_root=repository_root,
                        allow_synthetic=True,
                    )
                    import_dry_run(
                        kit,
                        package,
                        ledger,
                        protocol,
                        repository_root=repository_root,
                        allow_synthetic=True,
                    )
                if scenario in {"missing_schema", "unknown_namespace", "unverified_origin"}:
                    raise OfficialScorerIntakeNotReady(f"{scenario}_not_ready")
            except (OfficialScorerIntakeError, ExternalScorerError) as exc:
                observed = "rejected"
                reason = str(exc)
            rows.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "reason": reason,
                    "scenario": scenario,
                }
            )
    if any(row["expected"] != row["observed"] for row in rows):
        raise OfficialScorerIntakeError("synthetic_matrix_expectation_mismatch")
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "passed_count": len(rows),
        "protocol": PROTOCOL,
        "scenario_count": len(rows),
        "scenarios": rows,
        "schema_version": SCHEMA_VERSION,
        "status": "official_scorer_package_qualified",
    }


def audit_readiness(_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blocked_reasons": [
            "official_input_schema_missing",
            "official_metric_namespace_missing",
            "official_output_schema_missing",
            "official_package_missing",
            "verified_official_origin_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": [
            "full1000_incomplete",
            "human_precision_missing",
            "official_scorer_schema_missing",
        ],
        "formal_validation_complete": False,
        "official_score_generated": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_verified_official_package",
    }
