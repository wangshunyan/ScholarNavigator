"""Deterministic public API/CLI/artifact compatibility governance."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from scholar_agent.app.api.routes import router
from scholar_agent.core.api_schemas import CostReport, HealthResponse
from scholar_agent.evaluation.run_provenance import RunManifestV1


PROTOCOL = "public_contract_compatibility_v1"
SCHEMA_VERSION = "1"
EXIT_COMPATIBLE = 0
EXIT_BREAKING = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ROOT = Path(__file__).resolve().parents[3]
_DROP_KEYS = {"description", "examples", "example", "externalDocs"}
_DECLARATION = re.compile(r"export\s+(?:interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)")
_SNAPSHOT_REQUIRED = {
    "artifacts",
    "cli",
    "content_sha256",
    "documentation",
    "execution",
    "extension_policies",
    "formal_validation_complete",
    "frontend",
    "frontend_openapi_consistency",
    "openapi",
    "protocol",
    "schema_version",
    "source_commit",
    "version_governance",
}


class ContractError(RuntimeError):
    pass


class ContractNotReady(ContractError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", "..", ".env", "third_party"} for part in pure.parts):
        raise ContractError("unsafe_contract_path")
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError("contract_path_escape") from exc
    return path


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ContractError("contract_json_duplicate_key")
        value[key] = child
    return value


def _reject_constant(_: str) -> Any:
    raise ContractError("contract_json_non_finite_number")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except FileNotFoundError as exc:
        raise ContractNotReady("contract_baseline_missing") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("contract_json_invalid") from exc
    if not isinstance(value, dict):
        raise ContractError("contract_root_not_object")
    return value


def parse_json_text(text: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContractError("contract_json_invalid") from exc
    if not isinstance(value, dict):
        raise ContractError("contract_root_not_object")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    required = {
        "artifact_contracts",
        "cli_contracts",
        "documentation_files",
        "execution",
        "extension_policies",
        "formal_validation_complete",
        "frontend_types",
        "protocol",
        "schema_version",
        "source_commit",
        "version_governance",
    }
    if set(value) != required or value.get("protocol") != PROTOCOL or value.get("schema_version") != SCHEMA_VERSION or value.get("formal_validation_complete") is not False:
        raise ContractError("protocol_schema_invalid")
    if value.get("execution") != {"gold_or_qrels_loaded": False, "llm_request_count": 0, "network_request_count": 0, "quality_metric_count": 0, "snapshot_write_count": 0}:
        raise ContractError("offline_boundary_drift")
    governance = value.get("version_governance")
    if governance != {
        "current_write_version": "1",
        "migration_registry": {},
        "supported_read_versions": ["1"],
    }:
        raise ContractError("version_governance_drift")
    if not isinstance(value.get("extension_policies"), dict):
        raise ContractError("extension_policy_invalid")
    return value


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _normalize(child) for key, child in sorted(value.items()) if key not in _DROP_KEYS}
    if isinstance(value, list):
        normalized = [_normalize(child) for child in value]
        if all(isinstance(child, str) for child in normalized):
            return sorted(set(normalized))
        return normalized
    if isinstance(value, tuple):
        return _normalize(list(value))
    return value


def _openapi_contract() -> dict[str, Any]:
    app = FastAPI(title="contract-snapshot", version="0")
    app.include_router(router)
    schema = _normalize(app.openapi())
    return {"components": schema.get("components", {}), "paths": schema.get("paths", {})}


def _extract_ts_declarations(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    declarations: dict[str, str] = {}
    for match in _DECLARATION.finditer(text):
        name = match.group(1)
        start = match.start()
        brace = text.find("{", match.end())
        equals = text.find("=", match.end())
        if brace != -1 and (equals == -1 or brace < equals):
            depth = 0
            end = None
            for index in range(brace, len(text)):
                if text[index] == "{": depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None: raise ContractError("frontend_declaration_unclosed")
        else:
            end = text.find(";", match.end())
            if end == -1: raise ContractError("frontend_declaration_unclosed")
            end += 1
        declaration = re.sub(r"\s+", " ", text[start:end]).strip()
        if name in declarations: raise ContractError("duplicate_frontend_type")
        declarations[name] = declaration
    if not declarations: raise ContractError("frontend_types_missing")
    return dict(sorted(declarations.items()))


def _canonical_ts_type(value: str) -> tuple[str, bool]:
    compact = re.sub(r"\s+", "", value)
    nullable = "null" in compact.split("|")
    compact = "|".join(part for part in compact.split("|") if part not in {"", "null"})
    if compact.startswith("Array<") and compact.endswith(">"):
        return f"array<{_canonical_ts_type(compact[6:-1])[0]}>", nullable
    if compact.endswith("[]"):
        return f"array<{_canonical_ts_type(compact[:-2])[0]}>", nullable
    if compact in {"string", "boolean"}:
        return compact, nullable
    if compact == "number":
        return "number", nullable
    if compact.startswith("{"):
        return "object", nullable
    if re.fullmatch(r'"[^"|]+"(?:\|"[^"|]+")+', compact):
        return "string", nullable
    return f"ref:{compact}", nullable


def _extract_ts_alias_types(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    aliases: dict[str, str] = {}
    for match in re.finditer(r"export\s+type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", text):
        end = text.find(";", match.end())
        if end == -1:
            raise ContractError("frontend_declaration_unclosed")
        aliases[match.group(1)] = _canonical_ts_type(text[match.end() : end])[0]
    return dict(sorted(aliases.items()))


def _extract_ts_interface_contracts(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    interfaces: dict[str, dict[str, dict[str, Any]]] = {}
    for match in re.finditer(r"export\s+interface\s+([A-Za-z_][A-Za-z0-9_]*)\s*{", text):
        name = match.group(1)
        depth = 1
        buffer = ""
        fields: dict[str, dict[str, Any]] = {}
        for character in text[match.end() :]:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    break
            buffer += character
            if character == ";" and depth == 1:
                field_match = re.match(
                    r"\s*([A-Za-z_][A-Za-z0-9_]*)(\?)?\s*:\s*(.*?)\s*;\s*$",
                    buffer,
                    flags=re.DOTALL,
                )
                if field_match:
                    if field_match.group(1) in fields:
                        raise ContractError("duplicate_frontend_field")
                    field_type, nullable = _canonical_ts_type(field_match.group(3))
                    fields[field_match.group(1)] = {
                        "nullable": nullable,
                        "required": field_match.group(2) is None,
                        "type": field_type,
                    }
                buffer = ""
        if depth != 0:
            raise ContractError("frontend_declaration_unclosed")
        interfaces[name] = dict(sorted(fields.items()))
    return dict(sorted(interfaces.items()))


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return "<dynamic>"


def _cli_ast(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    commands: set[str] = set()
    arguments: list[dict[str, Any]] = []
    for loop in (node for node in ast.walk(tree) if isinstance(node, ast.For)):
        if not isinstance(loop.target, ast.Name):
            continue
        values = _literal(loop.iter)
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            continue
        for call in (node for node in ast.walk(loop) if isinstance(node, ast.Call)):
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_parser"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == loop.target.id
            ):
                commands.update(values)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute): continue
        if node.func.attr == "add_parser" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            commands.add(node.args[0].value)
        if node.func.attr == "add_argument":
            names = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if names:
                keywords = {item.arg: _normalize(_literal(item.value)) for item in node.keywords if item.arg in {"action", "choices", "default", "dest", "required"}}
                arguments.append({"names": sorted(names), "options": keywords})
                if names == ["command"] and isinstance(keywords.get("choices"), list):
                    commands.update(
                        item for item in keywords["choices"] if isinstance(item, str)
                    )
    arguments.sort(key=lambda row: canonical_json(row))
    return {"arguments": arguments, "commands": sorted(commands)}


def _run_cli_once(path: Path, arguments: list[str], root: Path) -> tuple[int, bytes, bytes]:
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(path), *arguments],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError("cli_probe_unavailable") from exc
    return completed.returncode, completed.stdout, completed.stderr


def _single_cli_probe(
    path: Path,
    probe: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    arguments = probe.get("arguments")
    expected_exit = probe.get("expected_exit_code")
    command = probe.get("command")
    probe_id = probe.get("probe_id")
    if (
        not isinstance(arguments, list)
        or not all(isinstance(item, str) for item in arguments)
        or not isinstance(expected_exit, int)
        or not isinstance(command, str)
        or not isinstance(probe_id, str)
    ):
        raise ContractError("cli_probe_schema_invalid")
    runs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix="public-contract-cli-") as temporary:
            materialized = [item.replace("{temp}", temporary) for item in arguments]
            runs.append(_run_cli_once(path, materialized, root))
    first, second = runs
    if first != second:
        raise ContractError("cli_probe_nondeterministic")
    exit_code, stdout, stderr = first
    if stderr:
        raise ContractError("cli_probe_stderr_not_empty")
    if exit_code != expected_exit:
        raise ContractError("cli_probe_exit_code_drift")
    try:
        payload = parse_json_text(stdout.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ContractError("cli_probe_utf8_invalid") from exc
    return {
        "arguments": arguments,
        "command": command,
        "exit_code": exit_code,
        "machine_output_schema": recursive_schema(payload),
        "probe_id": probe_id,
        "top_level_fields": sorted(payload),
    }


def _cli_probes(path: Path, spec: Mapping[str, Any], root: Path) -> dict[str, Any]:
    probes = spec.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ContractError("cli_probe_missing")
    results: dict[str, Any] = {}
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise ContractError("cli_probe_schema_invalid")
        result = _single_cli_probe(path, probe, root)
        probe_id = result["probe_id"]
        if probe_id in results:
            raise ContractError("cli_probe_duplicate")
        results[probe_id] = result
    return dict(sorted(results.items()))


def _schema_union(values: list[dict[str, Any]]) -> dict[str, Any]:
    unique = {stable_hash(value): value for value in values}
    ordered = [unique[key] for key in sorted(unique)]
    return ordered[0] if len(ordered) == 1 else {"anyOf": ordered}


def recursive_schema(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {
            "items": _schema_union([recursive_schema(child) for child in value]) if value else {},
            "type": "array",
        }
    if isinstance(value, Mapping):
        properties = {key: recursive_schema(child) for key, child in sorted(value.items())}
        return {
            "additionalProperties": False,
            "properties": properties,
            "required": sorted(properties),
            "type": "object",
        }
    raise ContractError("unsupported_artifact_value_type")


def _artifact_contract(path: Path) -> dict[str, Any]:
    value = load_json(path)
    versions = {key: value[key] for key in ("schema_version", "protocol", "contract", "analysis") if key in value and isinstance(value[key], str)}
    return {
        "recursive_schema": recursive_schema(value),
        "strict_unknown_fields": True,
        "versions": versions,
    }


def _schema_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
            references.add(reference.rsplit("/", 1)[-1])
        for child in value.values():
            references.update(_schema_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_schema_references(child))
    return references


def _directional_models(openapi: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    schemas = openapi.get("components", {}).get("schemas", {})
    request_models: set[str] = set()
    response_models: set[str] = set()
    for path in openapi.get("paths", {}).values():
        if not isinstance(path, Mapping):
            continue
        for operation in path.values():
            if not isinstance(operation, Mapping):
                continue
            request_models.update(_schema_references(operation.get("requestBody", {})))
            response_models.update(_schema_references(operation.get("responses", {})))
    for collection in (request_models, response_models):
        pending = list(collection)
        while pending:
            name = pending.pop()
            for reference in _schema_references(schemas.get(name, {})) - collection:
                collection.add(reference)
                pending.append(reference)
    return request_models, response_models


def _openapi_type(
    value: Mapping[str, Any],
    schemas: Mapping[str, Any] | None = None,
) -> tuple[str, bool]:
    if "$ref" in value:
        name = str(value["$ref"]).rsplit("/", 1)[-1]
        target = schemas.get(name) if schemas else None
        if isinstance(target, Mapping):
            field_type, nullable = _openapi_type(target, schemas)
            return field_type, nullable
        return f"ref:{name}", False
    variants = value.get("anyOf")
    if isinstance(variants, list):
        nullable = any(isinstance(item, Mapping) and item.get("type") == "null" for item in variants)
        concrete = [item for item in variants if isinstance(item, Mapping) and item.get("type") != "null"]
        if len(concrete) == 1:
            field_type, _ = _openapi_type(concrete[0], schemas)
            return field_type, nullable
    kind = value.get("type")
    if kind == "array":
        item_type, _ = _openapi_type(value.get("items", {}), schemas)
        return f"array<{item_type}>", False
    if kind in {"integer", "number"}:
        return "number", False
    if kind in {"string", "boolean", "object"}:
        return str(kind), False
    if isinstance(value.get("enum"), list):
        return "string_enum", False
    return "unknown", False


def _resolve_frontend_type(
    value: str,
    interfaces: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> str:
    if value.startswith("array<") and value.endswith(">"):
        return f"array<{_resolve_frontend_type(value[6:-1], interfaces, aliases)}>"
    if value.startswith("ref:"):
        name = value[4:]
        if name in interfaces or name.startswith("Record<"):
            return "object"
        if name in aliases:
            return _resolve_frontend_type(aliases[name], interfaces, aliases)
    return value


def _serialization_fixtures() -> dict[str, Any]:
    fixtures = {
        "CostReport": CostReport(),
        "HealthResponse": HealthResponse(
            version="fixture",
            time=datetime(2000, 1, 1, tzinfo=timezone.utc),
        ),
    }
    return {
        name: {
            "present_fields": sorted(jsonable_encoder(model)),
            "wire_schema": recursive_schema(jsonable_encoder(model)),
        }
        for name, model in sorted(fixtures.items())
    }


def _frontend_openapi_consistency(
    openapi: Mapping[str, Any],
    frontend_contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    frontend_aliases: Mapping[str, str],
) -> dict[str, Any]:
    schemas = openapi.get("components", {}).get("schemas", {})
    if not isinstance(schemas, Mapping):
        raise ContractError("openapi_components_invalid")
    request_models, response_models = _directional_models(openapi)
    checked: dict[str, Any] = {}
    for name in sorted(set(frontend_contracts) & set(schemas)):
        schema = schemas[name]
        if not isinstance(schema, Mapping) or not isinstance(schema.get("properties", {}), Mapping):
            continue
        backend_properties = schema.get("properties", {})
        frontend_fields = frontend_contracts[name]
        if sorted(backend_properties) != sorted(frontend_fields):
            raise ContractError(f"frontend_openapi_field_mismatch:{name}")
        directions = [direction for direction, names in (("request", request_models), ("response", response_models)) if name in names]
        fields: dict[str, Any] = {}
        required = set(schema.get("required", []))
        for field_name, backend_schema in sorted(backend_properties.items()):
            if not isinstance(backend_schema, Mapping):
                raise ContractError(f"openapi_field_schema_invalid:{name}:{field_name}")
            backend_type, backend_nullable = _openapi_type(backend_schema, schemas)
            frontend = frontend_fields[field_name]
            frontend_type = _resolve_frontend_type(
                str(frontend["type"]), frontend_contracts, frontend_aliases
            )
            if backend_type != frontend_type:
                raise ContractError(f"frontend_openapi_type_mismatch:{name}:{field_name}")
            for direction in directions:
                if direction == "response":
                    if frontend["required"] and field_name not in backend_properties:
                        raise ContractError(f"frontend_response_presence_mismatch:{name}:{field_name}")
                    if not frontend["nullable"] and backend_nullable:
                        raise ContractError(f"frontend_response_nullability_mismatch:{name}:{field_name}")
                else:
                    if not frontend["required"] and field_name in required:
                        raise ContractError(f"frontend_request_required_mismatch:{name}:{field_name}")
                    if frontend["nullable"] and not backend_nullable:
                        raise ContractError(f"frontend_request_nullability_mismatch:{name}:{field_name}")
            fields[field_name] = {
                "backend_nullable": backend_nullable,
                "backend_schema_required": field_name in required,
                "frontend_nullable": frontend["nullable"],
                "frontend_required": frontend["required"],
                "type": backend_type,
                "wire_present_on_response": name in response_models,
            }
        checked[name] = {"directions": directions, "fields": fields}
    if not checked:
        raise ContractError("frontend_openapi_models_missing")
    return {
        "models": checked,
        "response_serialization_fixtures": _serialization_fixtures(),
        "semantics": {
            "request": "frontend_producer_must_fit_backend_consumer",
            "response": "backend_producer_must_fit_frontend_consumer; defaults_are_wire_present",
        },
    }


def build_snapshot(protocol: Mapping[str, Any], *, repository_root: Path = ROOT) -> dict[str, Any]:
    root = repository_root.resolve()
    cli = {}
    for name, spec in sorted(protocol["cli_contracts"].items()):
        cli_path = _safe(root, spec["path"])
        row = _cli_ast(cli_path)
        row["exit_codes"] = spec["exit_codes"]
        row["probes"] = _cli_probes(cli_path, spec, root)
        if set(row["commands"]) != {
            probe["command"] for probe in row["probes"].values()
        }:
            raise ContractError("cli_subcommand_probe_coverage_incomplete")
        cli[name] = row
    artifacts = {name: _artifact_contract(_safe(root, path)) for name, path in sorted(protocol["artifact_contracts"].items())}
    artifacts["run_manifest_v1"] = {"json_schema": _normalize(RunManifestV1.model_json_schema()), "versions": {"schema_version": "1"}}
    documents = {}
    for relative in protocol["documentation_files"]:
        path = _safe(root, relative)
        relevant = sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if "scripts/check_" in line or "_v1" in line)
        documents[relative] = stable_hash(relevant)
    frontend_path = _safe(root, protocol["frontend_types"])
    openapi = _openapi_contract()
    frontend_contracts = _extract_ts_interface_contracts(frontend_path)
    frontend_aliases = _extract_ts_alias_types(frontend_path)
    snapshot = {
        "artifacts": artifacts,
        "cli": cli,
        "documentation": documents,
        "execution": dict(protocol["execution"]),
        "extension_policies": dict(protocol["extension_policies"]),
        "formal_validation_complete": False,
        "frontend": {"declarations": _extract_ts_declarations(frontend_path)},
        "frontend_openapi_consistency": _frontend_openapi_consistency(
            openapi, frontend_contracts, frontend_aliases
        ),
        "openapi": openapi,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": protocol["source_commit"],
        "version_governance": dict(protocol["version_governance"]),
    }
    snapshot["content_sha256"] = stable_hash(snapshot)
    return snapshot


def _at_path(value: Any, path: str) -> Any:
    current = value
    for part in path.removeprefix("$").strip("/").split("/"):
        if not part:
            continue
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _addition_classification(
    base_root: Mapping[str, Any],
    current_root: Mapping[str, Any],
    parent_path: str,
    key: str,
) -> str:
    policies = base_root.get("extension_policies", {})
    pointer = parent_path.removeprefix("$") or "/"
    if not isinstance(policies, Mapping) or policies.get(pointer) != "allow_optional":
        return "breaking"
    if parent_path.endswith("/properties"):
        schema_path = parent_path[: -len("/properties")]
        baseline_schema = _at_path(base_root, schema_path)
        current_schema = _at_path(current_root, schema_path)
        if not isinstance(baseline_schema, Mapping) or not isinstance(current_schema, Mapping):
            return "breaking"
        if baseline_schema.get("additionalProperties") is False:
            return "breaking"
        required = current_schema.get("required", [])
        if not isinstance(required, list) or key in required:
            return "breaking"
        return "additive_review_required"
    return "breaking"


def _diff(
    base: Any,
    current: Any,
    path: str,
    changes: list[dict[str, str]],
    base_root: Mapping[str, Any],
    current_root: Mapping[str, Any],
) -> None:
    if type(base) is not type(current):
        changes.append({"classification": "breaking", "path": path, "reason": "type_changed"})
        return
    if isinstance(base, dict):
        for key in sorted(set(base) - set(current)):
            changes.append({"classification": "breaking", "path": f"{path}/{key}", "reason": "field_removed"})
        for key in sorted(set(current) - set(base)):
            classification = _addition_classification(base_root, current_root, path, key)
            changes.append({"classification": classification, "path": f"{path}/{key}", "reason": "field_added"})
        for key in sorted(set(base) & set(current)):
            _diff(base[key], current[key], f"{path}/{key}", changes, base_root, current_root)
    elif isinstance(base, list):
        if path.endswith("/enum") and not set(base).issubset(set(current)):
            changes.append({"classification": "breaking", "path": path, "reason": "enum_narrowed"})
        elif base != current:
            changes.append({"classification": "breaking", "path": path, "reason": "ordered_or_membership_changed"})
    elif base != current:
        changes.append({"classification": "breaking", "path": path, "reason": "value_changed"})


def compare_snapshots(
    base: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    extension_policy: str | None = None,
) -> dict[str, Any]:
    """Compare snapshots; legacy extension_policy cannot override baseline policy."""
    left = {key: value for key, value in base.items() if key != "content_sha256"}
    right = {key: value for key, value in current.items() if key != "content_sha256"}
    changes: list[dict[str, str]] = []
    _diff(left, right, "$", changes, left, right)
    classes = {row["classification"] for row in changes}
    classification = "breaking" if "breaking" in classes else "additive_review_required" if changes else "compatible"
    return {"change_count": len(changes), "changes": changes, "classification": classification, "protocol": PROTOCOL, "schema_version": SCHEMA_VERSION}


def validate_snapshot(
    value: Mapping[str, Any],
    *,
    supported_versions: set[str] | None = None,
) -> None:
    versions = supported_versions or {SCHEMA_VERSION}
    if set(value) != _SNAPSHOT_REQUIRED or value.get("protocol") != PROTOCOL or value.get("schema_version") not in versions:
        raise ContractError("contract_snapshot_schema_invalid")
    expected = stable_hash({key: child for key, child in value.items() if key != "content_sha256"})
    if value.get("content_sha256") != expected:
        raise ContractError("contract_snapshot_hash_mismatch")
    if value.get("formal_validation_complete") is not False:
        raise ContractError("formal_validation_boundary_drift")


def migrate_snapshot(
    value: Mapping[str, Any],
    *,
    target_version: str,
    migration_registry: Mapping[str, str],
) -> dict[str, Any]:
    source_version = value.get("schema_version")
    validate_snapshot(value, supported_versions={str(source_version)})
    if source_version == target_version:
        return dict(value)
    key = f"{source_version}->{target_version}"
    if migration_registry.get(key) != "v1_to_v2_envelope_v1" or key != "1->2":
        raise ContractError("contract_migration_missing")
    migrated = json.loads(canonical_json(value))
    migrated["schema_version"] = target_version
    migrated["version_governance"] = {
        "current_write_version": target_version,
        "migration_registry": {key: "v1_to_v2_envelope_v1"},
        "supported_read_versions": ["1", target_version],
    }
    migrated["content_sha256"] = stable_hash(
        {name: child for name, child in migrated.items() if name != "content_sha256"}
    )
    validate_snapshot(migrated, supported_versions={target_version})
    return migrated


def verify_current(protocol: Mapping[str, Any], baseline: Mapping[str, Any], *, repository_root: Path = ROOT) -> dict[str, Any]:
    validate_snapshot(baseline)
    current = build_snapshot(protocol, repository_root=repository_root)
    comparison = compare_snapshots(baseline, current)
    if comparison["classification"] != "compatible":
        raise ContractError("public_contract_drift")
    return {"baseline_sha256": baseline.get("content_sha256"), "contract_count": len(current["artifacts"]) + len(current["cli"]) + 2, "execution": protocol["execution"], "exit_code": 0, "formal_validation_complete": False, "protocol": PROTOCOL, "schema_version": SCHEMA_VERSION, "snapshot_sha256": current["content_sha256"], "status": "contracts_compatible"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    temporary.replace(path)
