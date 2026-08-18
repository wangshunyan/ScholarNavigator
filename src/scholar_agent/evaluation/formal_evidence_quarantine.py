"""Fail-closed quarantine for future formal evaluation evidence.

The module handles only versioned intake metadata and explicitly-authorized
evaluation/reporting reads.  It is not imported by retrieval runtime code and
does not inspect real labels, qrels, scorer output, or project configuration.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import shutil
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PROTOCOL = "formal_evidence_quarantine_v1"
INTAKE_CONTRACT = "formal_evidence_intake_manifest_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
_DIGEST_CHARS = frozenset("0123456789abcdef")
_READ_METHODS = {"read_text", "read_bytes", "open"}
_PROTOCOL_KEYS = {
    "allowed_consumer_prefixes",
    "evidence_types",
    "execution",
    "formal_validation_complete",
    "forbidden_consumer_roots",
    "forbidden_data_tokens",
    "forbidden_path_tokens",
    "forbidden_structured_keys",
    "intake_contract",
    "lifecycle_states",
    "posthoc_protected_components",
    "prohibited_uses",
    "protocol",
    "schema_version",
    "source_commit",
}
_EVIDENCE_TYPES = {
    "human_annotation_labels",
    "human_adjudication_result",
    "official_scorer_package",
    "official_scorer_output",
}
_LIFECYCLE_STATES = {
    "received",
    "validated",
    "locked",
    "reported",
    "stale_for_claim",
    "invalid",
}
_FROZEN_SECURITY_POLICY_FIELDS = (
    "source_commit",
    "allowed_consumer_prefixes",
    "evidence_types",
    "prohibited_uses",
    "lifecycle_states",
    "forbidden_consumer_roots",
    "forbidden_data_tokens",
    "forbidden_path_tokens",
    "forbidden_structured_keys",
    "posthoc_protected_components",
)
_FROZEN_SECURITY_POLICY_SHA256 = (
    "fea82ba0e6465524ef90df6181d4c5d1df056e838bab421a445fe0a33039f41b"
)


class QuarantineError(RuntimeError):
    """An intake or isolation invariant was violated."""


class QuarantineBlocked(QuarantineError):
    """Controls are valid but no real formal evidence is available."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _DIGEST_CHARS


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _DIGEST_CHARS


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise QuarantineError("unsafe_relative_path")
    if path.name == ".env" or path.parts[0] == "third_party":
        raise QuarantineError("prohibited_evidence_path")
    return path.as_posix()


class ArtifactIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> "ArtifactIdentity":
        _safe_relative(self.path)
        return self


class InputBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: str = Field(min_length=1, max_length=100)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Chronology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preregistration_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    intake_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    report_code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    proof: Literal["git_ancestry", "synthetic_fixture_only"]


class FormalEvidenceIntakeManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    contract: Literal["formal_evidence_intake_manifest_v1"] = INTAKE_CONTRACT
    intake_id: str = Field(pattern=r"^evidence:[0-9a-f]{64}$")
    evidence_type: Literal[
        "human_annotation_labels",
        "human_adjudication_result",
        "official_scorer_package",
        "official_scorer_output",
    ]
    evidence_protocol_version: str = Field(min_length=1, max_length=100)
    input_binding: InputBinding
    artifact: ArtifactIdentity
    allowed_consumers: list[str]
    prohibited_uses: list[str]
    lifecycle_state: Literal[
        "received", "validated", "locked", "reported", "stale_for_claim", "invalid"
    ]
    synthetic_only: bool
    chronology: Chronology
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_closed_manifest(self) -> "FormalEvidenceIntakeManifestV1":
        if self.allowed_consumers != sorted(set(self.allowed_consumers)):
            raise ValueError("allowed consumers must be sorted and unique")
        if self.prohibited_uses != sorted(set(self.prohibited_uses)):
            raise ValueError("prohibited uses must be sorted and unique")
        payload = self.model_dump(mode="json")
        claimed = payload.pop("manifest_sha256")
        if stable_hash(payload) != claimed:
            raise ValueError("intake manifest digest mismatch")
        expected_id = "evidence:" + stable_hash(
            {
                "artifact": self.artifact.model_dump(mode="json"),
                "evidence_type": self.evidence_type,
                "input_binding": self.input_binding.model_dump(mode="json"),
            }
        )
        if self.intake_id != expected_id:
            raise ValueError("intake identity mismatch")
        return self


def load_protocol(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineError("protocol_unavailable") from exc
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL or value.get("schema_version") != SCHEMA_VERSION:
        raise QuarantineError("protocol_version_invalid")
    try:
        if set(value) != _PROTOCOL_KEYS:
            raise ValueError
        if value["execution"] != EXECUTION or value["formal_validation_complete"] is not False:
            raise ValueError
        if not _is_commit(value["source_commit"]) or value["intake_contract"] != INTAKE_CONTRACT:
            raise ValueError
        if set(value["evidence_types"]) != _EVIDENCE_TYPES:
            raise ValueError
        if set(value["lifecycle_states"]) != _LIFECYCLE_STATES:
            raise ValueError
        for key in (
            "allowed_consumer_prefixes",
            "evidence_types",
            "forbidden_consumer_roots",
            "forbidden_data_tokens",
            "forbidden_path_tokens",
            "forbidden_structured_keys",
            "lifecycle_states",
            "prohibited_uses",
        ):
            items = value[key]
            if not isinstance(items, list) or not items or not all(isinstance(item, str) and item for item in items) or len(items) != len(set(items)):
                raise ValueError
        components = value["posthoc_protected_components"]
        if not isinstance(components, dict) or not components:
            raise ValueError
        for component, paths in components.items():
            if not isinstance(component, str) or not component or not isinstance(paths, list) or not paths or not all(isinstance(item, str) and item for item in paths):
                raise ValueError
            for registered_path in paths:
                _safe_relative(registered_path.rstrip("/"))
        for root in value["forbidden_consumer_roots"]:
            _safe_relative(root)
        frozen_policy = {
            field: value[field] for field in _FROZEN_SECURITY_POLICY_FIELDS
        }
        if stable_hash(frozen_policy) != _FROZEN_SECURITY_POLICY_SHA256:
            raise ValueError
    except (KeyError, TypeError, ValueError, QuarantineError) as exc:
        raise QuarantineError("protocol_schema_invalid") from exc
    return value


def build_intake_manifest(
    *,
    evidence_path: Path,
    evidence_root: Path,
    evidence_type: str,
    evidence_protocol_version: str,
    input_binding: Mapping[str, Any],
    chronology: Mapping[str, Any],
    protocol: Mapping[str, Any],
    synthetic_only: bool,
) -> FormalEvidenceIntakeManifestV1:
    root = evidence_root.resolve()
    path = evidence_path.resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise QuarantineError("evidence_outside_intake_root") from exc
    try:
        artifact = {
            "path": _safe_relative(relative),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    except (OSError, UnicodeError) as exc:
        raise QuarantineError("evidence_artifact_unavailable") from exc
    identity_payload = {"artifact": artifact, "evidence_type": evidence_type, "input_binding": dict(input_binding)}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": INTAKE_CONTRACT,
        "intake_id": "evidence:" + stable_hash(identity_payload),
        "evidence_type": evidence_type,
        "evidence_protocol_version": evidence_protocol_version,
        "input_binding": dict(input_binding),
        "artifact": artifact,
        "allowed_consumers": sorted(protocol["allowed_consumer_prefixes"]),
        "prohibited_uses": sorted(protocol["prohibited_uses"]),
        "lifecycle_state": "locked",
        "synthetic_only": synthetic_only,
        "chronology": dict(chronology),
    }
    payload["manifest_sha256"] = stable_hash(payload)
    try:
        manifest = FormalEvidenceIntakeManifestV1.model_validate(payload)
    except ValidationError as exc:
        raise QuarantineError("intake_manifest_invalid") from exc
    _validate_chronology(manifest, repository_root=None)
    return manifest


def load_intake_manifest(path: Path) -> FormalEvidenceIntakeManifestV1:
    try:
        return FormalEvidenceIntakeManifestV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise QuarantineError("intake_manifest_invalid") from exc


def verify_intake_manifest(
    manifest: FormalEvidenceIntakeManifestV1,
    *,
    evidence_root: Path,
    protocol: Mapping[str, Any],
    repository_root: Path | None = None,
) -> None:
    if manifest.evidence_type not in protocol["evidence_types"]:
        raise QuarantineError("evidence_type_not_registered")
    if manifest.allowed_consumers != sorted(protocol["allowed_consumer_prefixes"]):
        raise QuarantineError("allowed_consumer_drift")
    if manifest.prohibited_uses != sorted(protocol["prohibited_uses"]):
        raise QuarantineError("prohibited_use_drift")
    artifact = (evidence_root.resolve() / manifest.artifact.path).resolve()
    try:
        artifact.relative_to(evidence_root.resolve())
    except ValueError as exc:
        raise QuarantineError("evidence_path_escape") from exc
    if not artifact.is_file() or artifact.stat().st_size != manifest.artifact.size or sha256_file(artifact) != manifest.artifact.sha256:
        raise QuarantineError("evidence_artifact_hash_mismatch")
    _validate_chronology(manifest, repository_root=repository_root)


def _validate_chronology(manifest: FormalEvidenceIntakeManifestV1, repository_root: Path | None) -> None:
    chronology = manifest.chronology
    chain = [
        chronology.preregistration_commit,
        chronology.execution_commit,
        chronology.intake_commit,
        chronology.report_code_commit,
    ]
    if len(set(chain)) != len(chain):
        raise QuarantineError("evidence_chronology_not_strict")
    if chronology.proof == "synthetic_fixture_only":
        if not manifest.synthetic_only:
            raise QuarantineError("synthetic_chronology_cannot_prove_real_evidence")
        return
    if manifest.synthetic_only or repository_root is None:
        raise QuarantineError("git_chronology_proof_unavailable")
    for parent, child in zip(chain, chain[1:]):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", parent, child],
            cwd=repository_root,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            raise QuarantineError("evidence_chronology_invalid")


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root / "src").with_suffix("")
    if relative.name == "__init__":
        relative = relative.parent
    return ".".join(relative.parts)


def verify_boundaries(repository_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """AST-scan forbidden runtime roots for formal evidence dependencies."""

    root = repository_root.resolve()
    violations: list[dict[str, str]] = []
    tokens = tuple(str(item).casefold() for item in protocol["forbidden_data_tokens"])
    path_tokens = tuple(str(item).casefold() for item in protocol["forbidden_path_tokens"])
    structured_keys = frozenset(str(item).casefold() for item in protocol["forbidden_structured_keys"])
    protected_module = "scholar_agent.evaluation.formal_evidence_quarantine"
    source_root = root / "src"
    trees: dict[Path, ast.AST] = {}
    graph: dict[str, set[str]] = {}
    if source_root.is_dir():
        source_paths = sorted(source_root.rglob("*.py"))
        known_modules = {_module_name(root, path) for path in source_paths}
        for path in source_paths:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise QuarantineError("boundary_source_unreadable") from exc
            trees[path.resolve()] = tree
            imports: set[str] = set()
            module_name = _module_name(root, path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.update(
                        _resolve_import_from(
                            module_name, path.name == "__init__.py", node, known_modules
                        )
                    )
            graph[module_name] = imports
    runtime_modules: set[str] = set()
    runtime_module_paths: dict[str, str] = {}
    for relative_root in protocol["forbidden_consumer_roots"]:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*.py")):
            tree = trees.get(path.resolve())
            if tree is None:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
                except (OSError, UnicodeError, SyntaxError) as exc:
                    raise QuarantineError("boundary_source_unreadable") from exc
            if source_root in path.resolve().parents:
                runtime_module = _module_name(root, path)
                runtime_modules.add(runtime_module)
                runtime_module_paths[runtime_module] = path.relative_to(root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_reads_path(node):
                    for argument in node.args[:1]:
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and any(token in argument.value.casefold() for token in path_tokens):
                            violations.append({"path": path.relative_to(root).as_posix(), "invariant": "production_reads_formal_evidence", "symbol": stable_hash(argument.value)})
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    if node.slice.value.casefold() in structured_keys:
                        violations.append({"path": path.relative_to(root).as_posix(), "invariant": "formal_evidence_in_runtime_configuration", "symbol": stable_hash(node.slice.value)})
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.casefold() in structured_keys:
                            violations.append({"path": path.relative_to(root).as_posix(), "invariant": "formal_evidence_in_runtime_event", "symbol": stable_hash(key.value)})
                if isinstance(node, ast.keyword) and node.arg and node.arg.casefold() in structured_keys:
                    violations.append({"path": path.relative_to(root).as_posix(), "invariant": "formal_evidence_in_runtime_call", "symbol": stable_hash(node.arg)})
        for path in sorted(
            candidate
            for candidate in scan_root.rglob("*")
            if candidate.is_file() and candidate.suffix in {".js", ".json", ".jsx", ".ts", ".tsx"}
        ):
            try:
                content = path.read_text(encoding="utf-8").casefold()
            except (OSError, UnicodeError) as exc:
                raise QuarantineError("boundary_asset_unreadable") from exc
            for token in tokens:
                if token in content:
                    violations.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "invariant": "formal_evidence_token_in_runtime_asset",
                            "symbol": stable_hash(token),
                        }
                    )
    for module in sorted(runtime_modules):
        chain = _import_path(graph, module, protected_module)
        if chain:
            violations.append(
                {
                    "path": runtime_module_paths[module],
                    "invariant": "production_imports_formal_evidence",
                    "symbol": stable_hash(chain),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed" if not violations else "violation",
        "exit_code": EXIT_READY if not violations else EXIT_VIOLATION,
        "scanned_root_count": len(protocol["forbidden_consumer_roots"]),
        "violation_count": len(violations),
        "violations": sorted(violations, key=lambda row: (row["path"], row["invariant"], row["symbol"])),
        "execution": EXECUTION,
    }


def _import_path(graph: Mapping[str, set[str]], start: str, target: str) -> list[str]:
    pending: list[tuple[str, list[str]]] = [(start, [start])]
    visited: set[str] = set()
    while pending:
        module, chain = pending.pop(0)
        if module in visited:
            continue
        visited.add(module)
        for dependency in sorted(graph.get(module, set())):
            if dependency == target or dependency.startswith(target + "."):
                return [*chain, dependency]
            if dependency.startswith("scholar_agent.") and dependency not in visited:
                pending.append((dependency, [*chain, dependency]))
    return []


def _resolve_import_from(
    module: str,
    is_package: bool,
    node: ast.ImportFrom,
    known_modules: set[str],
) -> set[str]:
    if node.level == 0:
        resolved = str(node.module or "")
    else:
        package = module.split(".") if is_package else module.split(".")[:-1]
        remove = node.level - 1
        if remove > len(package):
            return set()
        base = package[: len(package) - remove] if remove else package
        if node.module:
            base.extend(node.module.split("."))
        resolved = ".".join(base)
    imports = {resolved} if resolved else set()
    for alias in node.names:
        candidate = f"{resolved}.{alias.name}" if resolved else alias.name
        # ``from package import name`` can import either an attribute or a
        # submodule.  Only retain the expanded candidate when a source module
        # with that identity exists, avoiding attribute false positives.
        if candidate in known_modules:
            imports.add(candidate)
    return imports


def _call_reads_path(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "open"
    return isinstance(node.func, ast.Attribute) and node.func.attr in _READ_METHODS


def _consumer_allowed(consumer: str, protocol: Mapping[str, Any]) -> bool:
    return any(consumer == prefix.rstrip(".") or consumer.startswith(prefix) for prefix in protocol["allowed_consumer_prefixes"])


@contextmanager
def quarantine_io_guard(*, artifact: Path) -> Iterator[None]:
    """Allow one exact read and reject copy/write/other reads during consumption."""

    allowed = artifact.resolve()
    original_open = builtins.open
    original_io_open = io.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        candidate = Path(file).resolve() if isinstance(file, (str, Path)) else None
        if candidate != allowed or any(flag in mode for flag in "wax+"):
            raise QuarantineError("quarantined_file_access_denied")
        return original_open(file, mode, *args, **kwargs)

    def guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        candidate = Path(file).resolve() if isinstance(file, (str, Path)) else None
        if candidate != allowed or any(flag in mode for flag in "wax+"):
            raise QuarantineError("quarantined_file_access_denied")
        return original_io_open(file, mode, *args, **kwargs)

    def deny_copy(*args: Any, **kwargs: Any) -> None:
        raise QuarantineError("quarantined_evidence_copy_denied")

    with patch("builtins.open", guarded_open), patch("io.open", guarded_io_open), patch("shutil.copy", deny_copy), patch("shutil.copy2", deny_copy), patch("shutil.copyfile", deny_copy):
        yield


def consume_for_evaluation(
    manifest: FormalEvidenceIntakeManifestV1,
    *,
    evidence_root: Path,
    consumer: str,
    purpose: Literal["evaluation", "reporting", "clearance"],
    protocol: Mapping[str, Any],
) -> bytes:
    if not _consumer_allowed(consumer, protocol):
        raise QuarantineError("consumer_not_allowed")
    if purpose not in {"evaluation", "reporting", "clearance"}:
        raise QuarantineError("purpose_not_allowed")
    if manifest.lifecycle_state not in {"locked", "reported"}:
        raise QuarantineError("evidence_lifecycle_not_consumable")
    verify_intake_manifest(manifest, evidence_root=evidence_root, protocol=protocol)
    artifact = evidence_root.resolve() / manifest.artifact.path
    with quarantine_io_guard(artifact=artifact):
        return artifact.read_bytes()


def audit_contamination(
    manifest: FormalEvidenceIntakeManifestV1,
    changed_paths: Sequence[str],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    affected: dict[str, list[str]] = {}
    for raw in sorted(set(changed_paths)):
        path = _safe_relative(raw)
        for component, registrations in sorted(protocol["posthoc_protected_components"].items()):
            if any(path == item or (item.endswith("/") and path.startswith(item)) for item in registrations):
                affected.setdefault(component, []).append(path)
    stale = bool(affected)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "stale_for_claim" if stale else "clean",
        "exit_code": EXIT_VIOLATION if stale else EXIT_READY,
        "intake_id": manifest.intake_id,
        "synthetic_only": manifest.synthetic_only,
        "affected_components": {key: sorted(value) for key, value in sorted(affected.items())},
        "minimum_rerun_components": sorted(affected),
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def current_readiness(repository_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    boundaries = verify_boundaries(repository_root, protocol)
    if boundaries["exit_code"] != EXIT_READY:
        return boundaries
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "blocked_no_real_formal_evidence",
        "exit_code": EXIT_BLOCKED,
        "controls_ready": True,
        "real_human_evidence_present": False,
        "real_official_scorer_evidence_present": False,
        "formal_validation_complete": False,
        "boundary_report_sha256": stable_hash(boundaries),
        "execution": EXECUTION,
    }
    return report


def synthetic_manifest(root: Path, protocol: Mapping[str, Any], *, evidence_type: str = "human_annotation_labels") -> FormalEvidenceIntakeManifestV1:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "intake" / "evidence.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(canonical_json({"fixture": "synthetic-only", "labels": []}))
    digest = "a" * 64
    return build_intake_manifest(
        evidence_path=artifact,
        evidence_root=root,
        evidence_type=evidence_type,
        evidence_protocol_version="synthetic-fixture-v1",
        input_binding={"contract": "comparison_plan_v1", "plan_sha256": digest, "run_manifest_sha256": digest, "query_order_sha256": digest},
        chronology={"preregistration_commit": "1" * 40, "execution_commit": "2" * 40, "intake_commit": "3" * 40, "report_code_commit": "4" * 40, "proof": "synthetic_fixture_only"},
        protocol=protocol,
        synthetic_only=True,
    )
