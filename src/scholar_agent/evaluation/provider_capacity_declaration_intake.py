"""Offline intake for provider-owned Full1000 capacity declarations.

The intake binds source-specific declarations to the frozen Full1000 request
manifest and provider pacing protocol.  It builds deterministic, standard
library-only kits, validates declarations without credentials, enforces
one-time challenges, and proves that imported capacity changes only pacing
timing.  It never calls a provider or starts the formal run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_provider_pacing import (
    CapacityDeclaration,
    CapacityProfile,
    SOURCES,
    execute_profile,
    load_operations,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "provider_capacity_declaration_intake_v1"
DECLARATION_PROTOCOL = "provider_capacity_declaration_v1"
KIT_PROTOCOL = "provider_capacity_declaration_kit_v1"
IMPORT_RECEIPT = "provider_capacity_import_receipt_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "a859a62569e9e3d488e61ecbccfb0656ccfefebd"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
FROZEN_PROTOCOL_SHA256 = (
    "63f6072d2e3a65253c308c9c1f6191fa0dcfe408b68488e73ec61edc5e9f5602"
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MAX_KIT_FILES = 8
MAX_KIT_MEMBER_BYTES = 2 * 1024 * 1024
MAX_KIT_BYTES = 8 * 1024 * 1024
CHALLENGE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
SOURCE_SCOPES = {
    "openalex": "openalex_polite_pool_optional",
    "arxiv": "public_anonymous",
    "semantic_scholar": "semantic_scholar_api_key_optional",
    "pubmed": "ncbi_api_key_optional",
}
EVIDENCE_TYPES = (
    "operator_verified_provider_contract_v1",
    "provider_owner_portal_export_v1",
    "provider_owner_structured_notice_v1",
)
RETRY_AFTER_SEMANTICS = (
    "delta_seconds_or_http_date_when_provider_supplies_header",
)
CAPACITY_MODELS = ("fixed_window_token_bucket_v1",)
EFFECTIVE_CONDITIONS = ("scope_alias_active_and_epoch_within_window",)
INVALIDATION_CONDITIONS = ("expiry_revocation_scope_or_binding_change",)
LIFECYCLE_STATES = ("active", "revoked")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class CapacityIntakeError(RuntimeError):
    """A declaration, kit, import, or immutable-request invariant failed."""


class CapacityIntakeNotReady(CapacityIntakeError):
    """Real declarations remain incomplete."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_KIT_MEMBER_BYTES:
            raise CapacityIntakeError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except CapacityIntakeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapacityIntakeError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise CapacityIntakeError("json_root_not_object")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
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
        raise CapacityIntakeError("unsafe_protocol_path")
    return value


def _protocol_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def _validate_bindings(root: Path, protocol: Mapping[str, Any]) -> None:
    expected = {
        "execution_plan",
        "launch_control",
        "pacing_protocol",
        "preregistration",
        "request_intents",
        "request_manifest",
    }
    bindings = protocol.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != expected:
        raise CapacityIntakeError("binding_inventory_invalid")
    for name, raw in sorted(bindings.items()):
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise CapacityIntakeError("binding_schema_invalid")
        relative = _safe_relative(str(raw["path"]))
        target = root / relative
        if not target.is_file():
            raise CapacityIntakeError(f"binding_missing:{name}")
        if sha256_file(target) != raw["sha256"]:
            raise CapacityIntakeError(f"binding_hash_drift:{name}")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    required = {
        "bindings",
        "capacity_contract",
        "execution",
        "formal_validation_complete",
        "intake_policy",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "source_scopes",
        "synthetic_scenarios",
    }
    if set(value) != required:
        raise CapacityIntakeError("protocol_schema_invalid")
    if value["protocol"] != PROTOCOL or value["schema_version"] != SCHEMA_VERSION:
        raise CapacityIntakeError("protocol_version_invalid")
    if value["source_commit"] != SOURCE_COMMIT:
        raise CapacityIntakeError("protocol_source_commit_invalid")
    if value["execution"] != EXECUTION_ZERO:
        raise CapacityIntakeError("offline_execution_contract_drift")
    if value["formal_validation_complete"] is not False:
        raise CapacityIntakeError("formal_validation_state_drift")
    if _protocol_digest(value) != value["protocol_sha256"]:
        raise CapacityIntakeError("protocol_digest_mismatch")
    if value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise CapacityIntakeError("protocol_content_drift")
    if value["population"] != {
        "http_attempt_upper": 19280,
        "logical_source_request_count": 9640,
        "query_count": 1000,
        "shard_count": 20,
        "sources": list(SOURCES),
    }:
        raise CapacityIntakeError("population_contract_drift")
    if value["source_scopes"] != SOURCE_SCOPES:
        raise CapacityIntakeError("source_scope_contract_drift")
    contract = value["capacity_contract"]
    if contract != {
        "capacity_models": list(CAPACITY_MODELS),
        "challenge_max_age_seconds": CHALLENGE_MAX_AGE_SECONDS,
        "declaration_protocol": DECLARATION_PROTOCOL,
        "evidence_types": list(EVIDENCE_TYPES),
        "effective_conditions": list(EFFECTIVE_CONDITIONS),
        "invalidation_conditions": list(INVALIDATION_CONDITIONS),
        "lifecycle_states": list(LIFECYCLE_STATES),
        "numeric_fields": [
            "requests_per_second",
            "requests_per_minute",
            "burst",
            "max_concurrency",
            "cooldown_seconds",
        ],
        "retry_after_semantics": list(RETRY_AFTER_SEMANTICS),
        "units": {
            "burst": "requests",
            "cooldown_seconds": "seconds",
            "max_concurrency": "requests",
            "requests_per_minute": "requests_per_minute",
            "requests_per_second": "requests_per_second",
        },
    }:
        raise CapacityIntakeError("capacity_contract_drift")
    policy = value["intake_policy"]
    if policy != {
        "challenge_consumption": "append_only_single_use_per_source",
        "declaration_authentication": (
            "hash_proves_content_integrity_not_declarant_identity"
        ),
        "forbidden_contents": [
            "absolute_path",
            "credential",
            "endpoint",
            "environment_value",
            "query_text",
            "request_header",
            "url_parameter",
        ],
        "import_effect": "sending_timing_and_concurrency_only",
        "real_readiness": "all_four_fresh_active_declarations_required",
        "unknown_policy": "not_available_fail_closed",
    }:
        raise CapacityIntakeError("intake_policy_drift")
    expected_scenarios = [
        "qualified_four_sources",
        "single_source_missing",
        "expired",
        "unit_mismatch",
        "burst_conflict",
        "zero_concurrency",
        "scope_drift",
        "challenge_replay",
        "declaration_tamper",
        "dynamic_reduction",
        "revoked",
    ]
    if value["synthetic_scenarios"] != {
        "names": expected_scenarios,
        "test_only": True,
    }:
        raise CapacityIntakeError("synthetic_scenario_contract_drift")
    _validate_bindings(repository_root, value)
    return value


def _exact_object(
    value: Any, expected: set[str], reason: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise CapacityIntakeError(reason)
    return value


def _validate_challenge(value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CapacityIntakeError("challenge_invalid")
    return value


def build_declaration_contract(
    protocol: Mapping[str, Any],
    *,
    source: str,
    challenge_id: str,
    issued_epoch: int,
) -> dict[str, Any]:
    if source not in SOURCES:
        raise CapacityIntakeError("source_invalid")
    _validate_challenge(challenge_id)
    if (
        isinstance(issued_epoch, bool)
        or not isinstance(issued_epoch, int)
        or issued_epoch < 0
    ):
        raise CapacityIntakeError("challenge_epoch_invalid")
    value: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source": source,
        "api_scope_alias": SOURCE_SCOPES[source],
        "challenge": {
            "challenge_id": challenge_id,
            "issued_epoch": issued_epoch,
            "max_age_seconds": CHALLENGE_MAX_AGE_SECONDS,
            "one_time": True,
        },
        "bindings": {
            name: protocol["bindings"][name]["sha256"]
            for name in (
                "execution_plan",
                "request_manifest",
                "pacing_protocol",
            )
        },
        "allowed_evidence_types": list(EVIDENCE_TYPES),
        "allowed_capacity_models": list(CAPACITY_MODELS),
        "allowed_effective_conditions": list(EFFECTIVE_CONDITIONS),
        "allowed_invalidation_conditions": list(INVALIDATION_CONDITIONS),
        "allowed_retry_after_semantics": list(RETRY_AFTER_SEMANTICS),
        "allowed_lifecycle_states": list(LIFECYCLE_STATES),
        "units": copy.deepcopy(protocol["capacity_contract"]["units"]),
        "privacy": {
            "absolute_paths_allowed": False,
            "credentials_allowed": False,
            "endpoints_allowed": False,
            "free_text_allowed": False,
            "query_text_allowed": False,
            "request_headers_allowed": False,
            "url_parameters_allowed": False,
        },
        "formal_validation_complete": False,
    }
    value["contract_sha256"] = stable_hash(value)
    return value


def declaration_template(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": DECLARATION_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": contract["source_commit"],
        "source": contract["source"],
        "api_scope_alias": contract["api_scope_alias"],
        "challenge_id": contract["challenge"]["challenge_id"],
        "contract_sha256": contract["contract_sha256"],
        "capacity_model": CAPACITY_MODELS[0],
        "declaration_version": "replace_with_structured_version",
        "limits": {
            "requests_per_second": NOT_AVAILABLE,
            "requests_per_minute": NOT_AVAILABLE,
            "burst": NOT_AVAILABLE,
            "max_concurrency": NOT_AVAILABLE,
            "cooldown_seconds": NOT_AVAILABLE,
        },
        "units": copy.deepcopy(contract["units"]),
        "retry_after_semantics": RETRY_AFTER_SEMANTICS[0],
        "valid_from_epoch": NOT_AVAILABLE,
        "valid_until_epoch": NOT_AVAILABLE,
        "evidence_type": NOT_AVAILABLE,
        "effective_condition": EFFECTIVE_CONDITIONS[0],
        "invalidation_condition": INVALIDATION_CONDITIONS[0],
        "lifecycle_status": NOT_AVAILABLE,
        "supersedes_declaration_sha256": NOT_AVAILABLE,
        "synthetic_only": False,
        "declaration_sha256": "0" * 64,
    }


def _declaration_digest(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["declaration_sha256"] = "0" * 64
    return stable_hash(payload)


def seal_declaration(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["declaration_sha256"] = "0" * 64
    result["declaration_sha256"] = _declaration_digest(result)
    return result


def validate_declaration(
    contract: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    current_epoch: int,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    expected = {
        "api_scope_alias",
        "capacity_model",
        "challenge_id",
        "contract_sha256",
        "declaration_sha256",
        "declaration_version",
        "evidence_type",
        "effective_condition",
        "invalidation_condition",
        "lifecycle_status",
        "limits",
        "protocol",
        "retry_after_semantics",
        "schema_version",
        "source",
        "source_commit",
        "supersedes_declaration_sha256",
        "synthetic_only",
        "units",
        "valid_from_epoch",
        "valid_until_epoch",
    }
    declaration = _exact_object(value, expected, "declaration_schema_invalid")
    if (
        declaration["protocol"] != DECLARATION_PROTOCOL
        or declaration["schema_version"] != SCHEMA_VERSION
        or declaration["source_commit"] != contract["source_commit"]
        or declaration["source"] != contract["source"]
        or declaration["api_scope_alias"] != contract["api_scope_alias"]
        or declaration["challenge_id"] != contract["challenge"]["challenge_id"]
        or declaration["contract_sha256"] != contract["contract_sha256"]
    ):
        raise CapacityIntakeError("declaration_binding_invalid")
    if (
        not isinstance(declaration["declaration_version"], str)
        or not VERSION_RE.fullmatch(declaration["declaration_version"])
    ):
        raise CapacityIntakeError("declaration_version_invalid")
    if declaration["evidence_type"] not in contract["allowed_evidence_types"]:
        raise CapacityIntakeError("evidence_type_invalid")
    if declaration["capacity_model"] not in contract["allowed_capacity_models"]:
        raise CapacityIntakeError("capacity_model_invalid")
    if (
        declaration["effective_condition"]
        not in contract["allowed_effective_conditions"]
        or declaration["invalidation_condition"]
        not in contract["allowed_invalidation_conditions"]
    ):
        raise CapacityIntakeError("capacity_activation_condition_invalid")
    if (
        declaration["retry_after_semantics"]
        not in contract["allowed_retry_after_semantics"]
    ):
        raise CapacityIntakeError("retry_after_semantics_invalid")
    if declaration["lifecycle_status"] not in contract["allowed_lifecycle_states"]:
        raise CapacityIntakeError("lifecycle_status_invalid")
    if declaration["lifecycle_status"] != "active":
        raise CapacityIntakeError("declaration_revoked")
    if declaration["units"] != contract["units"]:
        raise CapacityIntakeError("capacity_unit_mismatch")
    limits = _exact_object(
        declaration["limits"],
        {
            "burst",
            "cooldown_seconds",
            "max_concurrency",
            "requests_per_minute",
            "requests_per_second",
        },
        "capacity_limits_schema_invalid",
    )
    for name, raw in limits.items():
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise CapacityIntakeError(f"capacity_value_invalid:{name}")
    if limits["burst"] < limits["max_concurrency"]:
        raise CapacityIntakeError("capacity_burst_below_concurrency")
    if limits["requests_per_minute"] < limits["requests_per_second"]:
        raise CapacityIntakeError("capacity_window_contradiction")
    valid_from = declaration["valid_from_epoch"]
    valid_until = declaration["valid_until_epoch"]
    if (
        isinstance(valid_from, bool)
        or not isinstance(valid_from, int)
        or isinstance(valid_until, bool)
        or not isinstance(valid_until, int)
        or valid_from < 0
        or valid_until <= valid_from
    ):
        raise CapacityIntakeError("capacity_validity_invalid")
    if (
        isinstance(current_epoch, bool)
        or not isinstance(current_epoch, int)
        or current_epoch < valid_from
        or current_epoch > valid_until
        or current_epoch
        > contract["challenge"]["issued_epoch"]
        + contract["challenge"]["max_age_seconds"]
    ):
        raise CapacityIntakeError("capacity_declaration_expired")
    supersedes = declaration["supersedes_declaration_sha256"]
    if supersedes != NOT_AVAILABLE and (
        not isinstance(supersedes, str) or not SHA256_RE.fullmatch(supersedes)
    ):
        raise CapacityIntakeError("supersession_identity_invalid")
    if declaration["synthetic_only"] is not False and not (
        allow_synthetic and declaration["synthetic_only"] is True
    ):
        raise CapacityIntakeError("synthetic_declaration_not_real")
    claimed = declaration["declaration_sha256"]
    if not isinstance(claimed, str) or _declaration_digest(declaration) != claimed:
        raise CapacityIntakeError("declaration_digest_invalid")
    return copy.deepcopy(declaration)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _manifest_self_hash(value: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(value))
    payload["manifest_self_sha256"] = "0" * 64
    return stable_hash(payload)


def build_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    source: str,
    challenge_id: str,
    issued_epoch: int,
    output: Path,
) -> dict[str, Any]:
    runtime = repository_root / "scripts/provider_capacity_declaration_runtime.py"
    contract = build_declaration_contract(
        protocol,
        source=source,
        challenge_id=challenge_id,
        issued_epoch=issued_epoch,
    )
    files = {
        "declaration_contract.json": canonical_json(contract),
        "declaration_template.json": canonical_json(
            declaration_template(contract)
        ),
        "verify.py": runtime.read_bytes(),
        "README.txt": (
            "Use a trusted copy of verify.py with python -I -S. Fill only "
            "structured numeric fields and listed enums in the declaration "
            "template. The kit contains no endpoint, credential, request "
            "header, URL parameter, query text, or repository dependency. "
            "A content hash does not authenticate the declarant.\n"
        ).encode("utf-8"),
    }
    inventory = [
        {"path": name, "sha256": sha256_bytes(payload), "size": len(payload)}
        for name, payload in sorted(files.items())
    ]
    manifest: dict[str, Any] = {
        "protocol": KIT_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source": source,
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "files": inventory,
        "manifest_self_sha256": "0" * 64,
    }
    manifest["manifest_self_sha256"] = _manifest_self_hash(manifest)
    files["manifest.json"] = canonical_json(manifest)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w") as archive:
            for name, payload in sorted(files.items()):
                archive.writestr(_zip_info(name), payload)
    except OSError as exc:
        raise CapacityIntakeError("kit_write_failed") from exc
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "capacity_declaration_kit_built",
        "exit_code": EXIT_READY,
        "source": source,
        "challenge_id": challenge_id,
        "contract_sha256": contract["contract_sha256"],
        "kit_sha256": sha256_file(output),
        "file_count": len(files),
        "execution": dict(EXECUTION_ZERO),
        "formal_validation_complete": False,
    }


def read_kit(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_KIT_BYTES:
            raise CapacityIntakeError("kit_size_or_presence_invalid")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_KIT_FILES:
                raise CapacityIntakeError("kit_file_limit")
            names: set[str] = set()
            files: dict[str, bytes] = {}
            for info in infos:
                name = info.filename
                pure = PurePosixPath(name)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    not name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or "\\" in name
                    or pure.as_posix() != name
                    or name in names
                    or mode not in (0, 0o100000)
                    or info.file_size > MAX_KIT_MEMBER_BYTES
                    or info.compress_size > MAX_KIT_MEMBER_BYTES
                ):
                    raise CapacityIntakeError("kit_member_unsafe")
                names.add(name)
                files[name] = archive.read(info)
    except CapacityIntakeError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CapacityIntakeError("kit_invalid") from exc
    expected = {
        "README.txt",
        "declaration_contract.json",
        "declaration_template.json",
        "manifest.json",
        "verify.py",
    }
    if set(files) != expected:
        raise CapacityIntakeError("kit_inventory_invalid")
    try:
        manifest = json.loads(
            files["manifest.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(token)
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapacityIntakeError("kit_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise CapacityIntakeError("kit_manifest_invalid")
    return manifest, files


def verify_kit(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    manifest, files = read_kit(path)
    if set(manifest) != {
        "challenge_id",
        "contract_sha256",
        "files",
        "manifest_self_sha256",
        "protocol",
        "schema_version",
        "source",
        "source_commit",
    }:
        raise CapacityIntakeError("kit_manifest_schema_invalid")
    if (
        manifest["protocol"] != KIT_PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["source"] not in SOURCES
        or _manifest_self_hash(manifest) != manifest["manifest_self_sha256"]
    ):
        raise CapacityIntakeError("kit_manifest_semantic_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise CapacityIntakeError("kit_inventory_schema_invalid")
    seen: set[str] = set()
    for raw in inventory:
        row = _exact_object(
            raw, {"path", "sha256", "size"}, "kit_inventory_entry_invalid"
        )
        name = row["path"]
        if (
            not isinstance(name, str)
            or name in seen
            or name == "manifest.json"
            or name not in files
            or not isinstance(row["size"], int)
            or isinstance(row["size"], bool)
            or row["size"] != len(files[name])
            or row["sha256"] != sha256_bytes(files[name])
        ):
            raise CapacityIntakeError("kit_inventory_entry_invalid")
        seen.add(name)
    if seen != set(files) - {"manifest.json"}:
        raise CapacityIntakeError("kit_inventory_not_closed")
    try:
        contract = json.loads(
            files["declaration_contract.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapacityIntakeError("kit_contract_invalid") from exc
    expected = build_declaration_contract(
        protocol,
        source=manifest["source"],
        challenge_id=manifest["challenge_id"],
        issued_epoch=contract.get("challenge", {}).get("issued_epoch", -1)
        if isinstance(contract, dict)
        else -1,
    )
    if contract != expected or contract["contract_sha256"] != manifest["contract_sha256"]:
        raise CapacityIntakeError("kit_contract_binding_invalid")
    if files["verify.py"] != (
        repository_root / "scripts/provider_capacity_declaration_runtime.py"
    ).read_bytes():
        raise CapacityIntakeError("kit_runtime_drift")
    if json.loads(files["declaration_template.json"]) != declaration_template(
        contract
    ):
        raise CapacityIntakeError("kit_template_drift")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "capacity_declaration_kit_verified",
        "exit_code": EXIT_READY,
        "source": manifest["source"],
        "challenge_id": manifest["challenge_id"],
        "kit_sha256": sha256_file(path),
        "execution": dict(EXECUTION_ZERO),
        "formal_validation_complete": False,
    }


def declaration_from_kit(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _manifest, files = read_kit(path)
    try:
        contract = json.loads(
            files["declaration_contract.json"].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CapacityIntakeError("kit_contract_invalid") from exc
    if not isinstance(contract, dict):
        raise CapacityIntakeError("kit_contract_invalid")
    return contract, json.loads(
        files["declaration_template.json"].decode("utf-8"),
        object_pairs_hook=_unique_object,
    )


def _empty_ledger() -> dict[str, Any]:
    return {
        "protocol": "provider_capacity_challenge_ledger_v1",
        "schema_version": SCHEMA_VERSION,
        "events": [],
        "ledger_sha256": stable_hash([]),
    }


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    value = read_object(path)
    if set(value) != {"events", "ledger_sha256", "protocol", "schema_version"}:
        raise CapacityIntakeError("challenge_ledger_schema_invalid")
    if (
        value["protocol"] != "provider_capacity_challenge_ledger_v1"
        or value["schema_version"] != SCHEMA_VERSION
        or not isinstance(value["events"], list)
        or stable_hash(value["events"]) != value["ledger_sha256"]
    ):
        raise CapacityIntakeError("challenge_ledger_invalid")
    previous = "0" * 64
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(value["events"]):
        row = _exact_object(
            raw,
            {
                "challenge_id",
                "declaration_sha256",
                "event_sha256",
                "previous_event_sha256",
                "sequence",
                "source",
            },
            "challenge_event_schema_invalid",
        )
        payload = dict(row)
        claimed = payload.pop("event_sha256")
        identity = (row["source"], row["challenge_id"])
        if (
            row["sequence"] != index
            or row["source"] not in SOURCES
            or identity in identities
            or row["previous_event_sha256"] != previous
            or stable_hash(payload) != claimed
        ):
            raise CapacityIntakeError("challenge_ledger_chain_invalid")
        identities.add(identity)
        previous = claimed
    return value


def _append_challenge_events(
    ledger: Mapping[str, Any], declarations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(ledger))
    events = result["events"]
    consumed = {(row["source"], row["challenge_id"]) for row in events}
    previous = events[-1]["event_sha256"] if events else "0" * 64
    for declaration in sorted(declarations, key=lambda row: SOURCES.index(row["source"])):
        identity = (declaration["source"], declaration["challenge_id"])
        if identity in consumed:
            raise CapacityIntakeError("challenge_replay")
        event = {
            "sequence": len(events),
            "source": declaration["source"],
            "challenge_id": declaration["challenge_id"],
            "declaration_sha256": declaration["declaration_sha256"],
            "previous_event_sha256": previous,
        }
        event["event_sha256"] = stable_hash(event)
        events.append(event)
        consumed.add(identity)
        previous = event["event_sha256"]
    result["ledger_sha256"] = stable_hash(events)
    return result


def _pacing_declarations(
    declarations: Sequence[Mapping[str, Any]],
) -> dict[str, CapacityDeclaration]:
    result: dict[str, CapacityDeclaration] = {}
    for value in declarations:
        limits = value["limits"]
        result[value["source"]] = CapacityDeclaration(
            source=value["source"],
            requests_per_second=limits["requests_per_second"],
            requests_per_minute=limits["requests_per_minute"],
            max_concurrency=limits["max_concurrency"],
            burst=limits["burst"],
            cooldown_steps=limits["cooldown_seconds"],
            declaration_version=value["declaration_version"],
            valid_from_step=0,
            valid_until_step=10_000_000,
        )
    return result


def prove_request_set_unchanged(
    repository_root: Path,
    pacing_protocol: Mapping[str, Any],
    declarations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    operations = load_operations(repository_root, pacing_protocol)
    before = {
        "intent_count": len(operations),
        "http_attempt_upper": sum(row.http_attempt_upper for row in operations),
        "shard_count": len({row.shard_index for row in operations}),
        "intent_order_sha256": stable_hash(
            [row.intent_identity for row in operations]
        ),
        "request_contract_sha256": stable_hash(
            [
                {
                    "intent": row.intent_identity,
                    "request_spec_sha256": row.request_spec_sha256,
                }
                for row in operations
            ]
        ),
    }
    machine = execute_profile(
        pacing_protocol,
        operations,
        CapacityProfile(
            "imported_capacity_declarations",
            _pacing_declarations(declarations),
        ),
    )
    summary = machine.summary()
    if (
        before["intent_count"] != 9640
        or before["http_attempt_upper"] != 19280
        or before["shard_count"] != 20
        or summary["intent_coverage_count"] != 9640
        or summary["request_parameter_mutation_count"] != 0
        or summary["request_set_unchanged"] is not True
        or summary["duplicate_request_count"] != 0
        or summary["window_violation_count"] != 0
        or summary["admitted_attempt_count"] > 19280
    ):
        raise CapacityIntakeError("request_set_or_budget_changed")
    return {
        **before,
        "admitted_attempt_count": summary["admitted_attempt_count"],
        "duplicate_request_count": summary["duplicate_request_count"],
        "window_violation_count": summary["window_violation_count"],
        "request_parameter_mutation_count": summary[
            "request_parameter_mutation_count"
        ],
        "request_set_unchanged": summary["request_set_unchanged"],
        "pacing_request_identity_sha256": summary["request_identity_sha256"],
        "pacing_request_contract_sha256": summary["request_contract_sha256"],
        "budget_conserved": summary["budget_conserved"],
    }


def import_declarations(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    entries: Mapping[str, tuple[Path, Path]],
    ledger_path: Path,
    current_epoch: int,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    if set(entries) != set(SOURCES):
        raise CapacityIntakeNotReady("missing_real_declarations")
    validated: list[dict[str, Any]] = []
    kit_hashes: dict[str, str] = {}
    for source in SOURCES:
        kit_path, declaration_path = entries[source]
        report = verify_kit(
            kit_path, protocol, repository_root=repository_root
        )
        if report["source"] != source:
            raise CapacityIntakeError("source_kit_mismatch")
        contract, _template = declaration_from_kit(kit_path)
        declaration = validate_declaration(
            contract,
            read_object(declaration_path),
            current_epoch=current_epoch,
            allow_synthetic=allow_synthetic,
        )
        if declaration["source"] != source:
            raise CapacityIntakeError("source_declaration_mismatch")
        validated.append(declaration)
        kit_hashes[source] = sha256_file(kit_path)
    ledger = _load_ledger(ledger_path)
    next_ledger = _append_challenge_events(ledger, validated)
    pacing_protocol = json.loads(
        (
            repository_root
            / protocol["bindings"]["pacing_protocol"]["path"]
        ).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    preservation = prove_request_set_unchanged(
        repository_root, pacing_protocol, validated
    )
    write_json(ledger_path, next_ledger)
    receipt: dict[str, Any] = {
        "protocol": IMPORT_RECEIPT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "intake_protocol_sha256": protocol["protocol_sha256"],
        "pacing_protocol_sha256": protocol["bindings"]["pacing_protocol"][
            "sha256"
        ],
        "launch_control_sha256": protocol["bindings"]["launch_control"][
            "sha256"
        ],
        "request_manifest_sha256": protocol["bindings"]["request_manifest"][
            "sha256"
        ],
        "declaration_sha256": {
            row["source"]: row["declaration_sha256"] for row in validated
        },
        "kit_sha256": kit_hashes,
        "challenge_ledger_sha256": next_ledger["ledger_sha256"],
        "request_preservation": preservation,
        "launch_activation_allowed": not any(
            row["synthetic_only"] for row in validated
        ),
        "synthetic_only": any(row["synthetic_only"] for row in validated),
        "formal_validation_complete": False,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def verify_import_receipt_for_launch(
    receipt: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    expected = {
        "challenge_ledger_sha256",
        "declaration_sha256",
        "formal_validation_complete",
        "intake_protocol_sha256",
        "kit_sha256",
        "launch_activation_allowed",
        "launch_control_sha256",
        "pacing_protocol_sha256",
        "protocol",
        "receipt_sha256",
        "request_manifest_sha256",
        "request_preservation",
        "schema_version",
        "source_commit",
        "synthetic_only",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise CapacityIntakeError("import_receipt_schema_invalid")
    payload = dict(receipt)
    claimed = payload.pop("receipt_sha256")
    if (
        receipt["protocol"] != IMPORT_RECEIPT
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["source_commit"] != SOURCE_COMMIT
        or receipt["intake_protocol_sha256"] != protocol["protocol_sha256"]
        or receipt["pacing_protocol_sha256"]
        != protocol["bindings"]["pacing_protocol"]["sha256"]
        or receipt["launch_control_sha256"]
        != protocol["bindings"]["launch_control"]["sha256"]
        or receipt["request_manifest_sha256"]
        != protocol["bindings"]["request_manifest"]["sha256"]
        or receipt["synthetic_only"] is not False
        or receipt["launch_activation_allowed"] is not True
        or receipt["formal_validation_complete"] is not False
        or stable_hash(payload) != claimed
    ):
        raise CapacityIntakeError("import_receipt_launch_binding_invalid")
    preservation = receipt["request_preservation"]
    if (
        not isinstance(preservation, dict)
        or preservation.get("intent_count") != 9640
        or preservation.get("http_attempt_upper") != 19280
        or preservation.get("shard_count") != 20
        or preservation.get("request_set_unchanged") is not True
        or preservation.get("request_parameter_mutation_count") != 0
        or preservation.get("duplicate_request_count") != 0
        or preservation.get("window_violation_count") != 0
        or preservation.get("budget_conserved") is not True
    ):
        raise CapacityIntakeError("import_receipt_request_preservation_invalid")


def _synthetic_declaration(
    contract: Mapping[str, Any],
    *,
    version: str = "synthetic-v1",
    current_epoch: int,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = declaration_template(contract)
    value.update(
        {
            "declaration_version": version,
            "limits": {
                "requests_per_second": 4,
                "requests_per_minute": 120,
                "burst": 8,
                "max_concurrency": 3,
                "cooldown_seconds": 5,
            },
            "valid_from_epoch": current_epoch - 1,
            "valid_until_epoch": current_epoch + 1000,
            "evidence_type": EVIDENCE_TYPES[0],
            "lifecycle_status": "active",
            "synthetic_only": True,
        }
    )
    if overrides:
        for key, replacement in overrides.items():
            if key.startswith("limits."):
                value["limits"][key.split(".", 1)[1]] = replacement
            else:
                value[key] = replacement
    return seal_declaration(value)


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    epoch = 1_700_000_100
    contracts = {
        source: build_declaration_contract(
            protocol,
            source=source,
            challenge_id=hashlib.sha256(
                f"capacity-intake:{source}".encode()
            ).hexdigest(),
            issued_epoch=1_700_000_000,
        )
        for source in SOURCES
    }
    declarations = {
        source: _synthetic_declaration(
            contracts[source], current_epoch=epoch
        )
        for source in SOURCES
    }
    rows: list[dict[str, Any]] = []
    preservation = prove_request_set_unchanged(
        repository_root,
        json.loads(
            (
                repository_root
                / protocol["bindings"]["pacing_protocol"]["path"]
            ).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        ),
        list(declarations.values()),
    )
    rows.append(
        {
            "scenario": "qualified_four_sources",
            "status": "accepted",
            "request_preservation": preservation,
        }
    )

    def rejected(
        name: str,
        source: str,
        overrides: Mapping[str, Any],
        reason: str,
    ) -> None:
        candidate = _synthetic_declaration(
            contracts[source], current_epoch=epoch, overrides=overrides
        )
        try:
            validate_declaration(
                contracts[source],
                candidate,
                current_epoch=epoch,
                allow_synthetic=True,
            )
        except CapacityIntakeError as exc:
            actual = str(exc).split(":", 1)[0]
            if actual != reason:
                raise CapacityIntakeError(
                    f"synthetic_expected_reason_mismatch:{name}"
                ) from exc
            rows.append(
                {"scenario": name, "status": "rejected", "reason_code": actual}
            )
            return
        raise CapacityIntakeError(f"synthetic_scenario_not_rejected:{name}")

    rows.append(
        {
            "scenario": "single_source_missing",
            "status": "not_ready",
            "reason_code": "missing_real_declarations",
        }
    )
    rejected(
        "expired",
        "openalex",
        {
            "valid_from_epoch": epoch - 100,
            "valid_until_epoch": epoch - 1,
        },
        "capacity_declaration_expired",
    )
    altered_units = copy.deepcopy(contracts["arxiv"]["units"])
    altered_units["requests_per_second"] = "requests_per_hour"
    rejected(
        "unit_mismatch",
        "arxiv",
        {"units": altered_units},
        "capacity_unit_mismatch",
    )
    rejected(
        "burst_conflict",
        "semantic_scholar",
        {"limits.burst": 1, "limits.max_concurrency": 3},
        "capacity_burst_below_concurrency",
    )
    rejected(
        "zero_concurrency",
        "pubmed",
        {"limits.max_concurrency": 0},
        "capacity_value_invalid",
    )
    rejected(
        "scope_drift",
        "openalex",
        {"api_scope_alias": SOURCE_SCOPES["arxiv"]},
        "declaration_binding_invalid",
    )
    replay_ledger = _append_challenge_events(
        _empty_ledger(), list(declarations.values())
    )
    try:
        _append_challenge_events(replay_ledger, [declarations["openalex"]])
    except CapacityIntakeError as exc:
        if str(exc) != "challenge_replay":
            raise
        rows.append(
            {
                "scenario": "challenge_replay",
                "status": "rejected",
                "reason_code": "challenge_replay",
            }
        )
    tampered = copy.deepcopy(declarations["arxiv"])
    tampered["limits"]["burst"] += 1
    try:
        validate_declaration(
            contracts["arxiv"],
            tampered,
            current_epoch=epoch,
            allow_synthetic=True,
        )
    except CapacityIntakeError as exc:
        if str(exc) != "declaration_digest_invalid":
            raise
        rows.append(
            {
                "scenario": "declaration_tamper",
                "status": "rejected",
                "reason_code": "declaration_digest_invalid",
            }
        )
    reduced = copy.deepcopy(declarations)
    reduced_openalex = _synthetic_declaration(
        contracts["openalex"],
        version="synthetic-reduced-v2",
        current_epoch=epoch,
        overrides={
            "limits.requests_per_second": 1,
            "limits.requests_per_minute": 30,
            "limits.burst": 3,
            "limits.max_concurrency": 1,
            "supersedes_declaration_sha256": declarations["openalex"][
                "declaration_sha256"
            ],
        },
    )
    reduced["openalex"] = reduced_openalex
    reduced_preservation = prove_request_set_unchanged(
        repository_root,
        json.loads(
            (
                repository_root
                / protocol["bindings"]["pacing_protocol"]["path"]
            ).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        ),
        list(reduced.values()),
    )
    rows.append(
        {
            "scenario": "dynamic_reduction",
            "status": "accepted",
            "request_preservation": reduced_preservation,
        }
    )
    rejected(
        "revoked",
        "pubmed",
        {"lifecycle_status": "revoked"},
        "declaration_revoked",
    )
    if [row["scenario"] for row in rows] != protocol["synthetic_scenarios"]["names"]:
        raise CapacityIntakeError("synthetic_scenario_order_drift")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "capacity_declarations_qualified",
        "exit_code": EXIT_READY,
        "scenario_count": len(rows),
        "accepted_scenario_count": sum(
            row["status"] == "accepted" for row in rows
        ),
        "rejected_or_blocked_scenario_count": sum(
            row["status"] != "accepted" for row in rows
        ),
        "scenarios": rows,
        "synthetic_only": True,
        "execution": dict(EXECUTION_ZERO),
        "formal_validation_complete": False,
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    missing = [
        {
            "source": source,
            "api_scope_alias": SOURCE_SCOPES[source],
            "status": "not_available",
        }
        for source in SOURCES
    ]
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_real_declarations",
        "exit_code": EXIT_NOT_READY,
        "activation_allowed": False,
        "missing_declaration_count": len(missing),
        "missing_declarations": missing,
        "missing_declarations_sha256": stable_hash(missing),
        "pacing_protocol_sha256": protocol["bindings"]["pacing_protocol"][
            "sha256"
        ],
        "launch_control_sha256": protocol["bindings"]["launch_control"][
            "sha256"
        ],
        "network_status": "not_checked",
        "execution": dict(EXECUTION_ZERO),
        "formal_validation_complete": False,
    }


def build_launch_addendum(
    protocol: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "contract": "full1000_provider_capacity_intake_addendum_v1",
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "intake_protocol_sha256": protocol["protocol_sha256"],
        "pacing_protocol_sha256": protocol["bindings"]["pacing_protocol"][
            "sha256"
        ],
        "launch_control_sha256": protocol["bindings"]["launch_control"][
            "sha256"
        ],
        "request_manifest_sha256": protocol["bindings"]["request_manifest"][
            "sha256"
        ],
        "logical_source_request_count": 9640,
        "http_attempt_upper": 19280,
        "shard_count": 20,
        "activation_requirement": (
            "all_four_fresh_active_declarations_and_single_use_import_receipt"
        ),
        "real_declaration_status": readiness["status"],
        "request_set_mutated": False,
        "network_status": "not_checked",
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    value["addendum_sha256"] = stable_hash(value)
    return value
