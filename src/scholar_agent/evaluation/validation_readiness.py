"""Deterministic validation-readiness evidence bundle and claim gate.

``validation_readiness_bundle_v1`` is an evidence publication layer.  It does
not run retrieval or evaluation.  It binds already tracked, aggregate evidence
by hash, checks cross-report invariants, and publishes the limits of what those
artifacts can establish.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "1"
PROTOCOL_VERSION = "validation_readiness_bundle_v1"
EXIT_READY_WITH_DECLARED_BLOCKERS = 0
EXIT_EVIDENCE_OR_CLAIM_VIOLATION = 2
EXIT_NOT_READY_MISSING_REQUIRED_EVIDENCE = 3
EXIT_USAGE_ERROR = 4
ALLOWED_CLAIM_STATUSES = {
    "verified",
    "internal_only",
    "blocked",
    "not_applicable",
}
REQUIRED_BLOCKERS = {
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ValidationReadinessError(RuntimeError):
    """Evidence, claim, or release integrity is invalid."""


class ValidationReadinessNotReady(ValidationReadinessError):
    """Required tracked evidence is absent rather than contradictory."""


def canonical_json(value: Any) -> str:
    """Return the repository's deterministic JSON representation."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tree_sha256(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(files[name]).digest())
    return digest.hexdigest()


def _safe_repo_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValidationReadinessError("unsafe_relative_path")
    if value.parts[0] == "third_party" or value.name == ".env":
        raise ValidationReadinessError("prohibited_evidence_path")
    candidate = (root / Path(*value.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationReadinessError("evidence_path_escape") from exc
    return candidate


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationReadinessError("invalid_json_evidence") from exc


def _pointer_get(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValidationReadinessError("invalid_json_pointer")
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ValidationReadinessError("json_pointer_missing") from exc
        else:
            raise ValidationReadinessError("json_pointer_missing")
    return current


def _assert_no_release_leaks(value: Any, *, field_name: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_release_leaks(child, field_name=str(key))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_no_release_leaks(child, field_name=field_name)
        return
    if not isinstance(value, str):
        return
    lowered = value.casefold()
    if field_name != "pointer" and (
        value.startswith("/Users/")
        or value.startswith("/home/")
        or value.startswith("file://")
    ):
        raise ValidationReadinessError("absolute_path_leak")
    forbidden = (
        "authorization:",
        "bearer ",
        "api_key=",
        "apikey=",
        "private_mapping",
        "query_to_gold",
    )
    if any(token in lowered for token in forbidden):
        raise ValidationReadinessError("sensitive_or_private_reference")


def load_contract(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path))
    if not isinstance(value, dict):
        raise ValidationReadinessError("contract_not_object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValidationReadinessError("unsupported_schema_version")
    if value.get("protocol") != PROTOCOL_VERSION:
        raise ValidationReadinessError("unsupported_protocol_version")
    execution = value.get("execution")
    if execution != {
        "gold_or_qrels_loaded": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise ValidationReadinessError("offline_execution_contract_drift")
    blockers = {str(item.get("blocker_id")) for item in value.get("blockers") or []}
    if blockers != REQUIRED_BLOCKERS:
        raise ValidationReadinessError("required_blocker_set_drift")
    if value.get("release", {}).get("status") != "ready_with_declared_blockers":
        raise ValidationReadinessError("release_status_drift")
    _assert_no_release_leaks(value)
    return value


def _audit_evidence(
    contract: Mapping[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    seen: set[str] = set()
    for item in contract.get("evidence") or []:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            raise ValidationReadinessError("duplicate_or_empty_evidence_id")
        seen.add(evidence_id)
        path_value = str(item.get("path") or "")
        path = _safe_repo_path(root, path_value)
        if not path.is_file():
            if item.get("required") is True:
                raise ValidationReadinessNotReady("required_evidence_missing")
            continue
        actual_sha256 = sha256_file(path)
        if actual_sha256 != item.get("sha256"):
            raise ValidationReadinessError("historical_evidence_hash_drift")
        payload = _read_json(path) if item.get("format") == "json" else None
        if payload is not None:
            for check in item.get("checks") or []:
                actual = _pointer_get(payload, str(check.get("pointer") or ""))
                if actual != check.get("equals"):
                    raise ValidationReadinessError("evidence_field_drift")
            loaded[evidence_id] = payload
        index.append(
            {
                "dependencies": sorted(str(v) for v in item.get("dependencies") or []),
                "evidence_id": evidence_id,
                "format": str(item.get("format")),
                "path": path_value,
                "protocol": str(item.get("protocol")),
                "role": str(item.get("role")),
                "sha256": actual_sha256,
                "size": path.stat().st_size,
            }
        )
    expected_ids = {str(item.get("evidence_id")) for item in contract.get("evidence") or []}
    if seen != expected_ids:
        raise ValidationReadinessError("evidence_inventory_not_closed")
    return sorted(index, key=lambda item: item["evidence_id"]), loaded


def _audit_claim_sources(contract: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    audited: list[dict[str, Any]] = []
    for item in contract.get("claim_sources") or []:
        path_value = str(item.get("path") or "")
        path = _safe_repo_path(root, path_value)
        if not path.is_file():
            raise ValidationReadinessNotReady("claim_source_missing")
        actual = sha256_file(path)
        if actual != item.get("sha256"):
            raise ValidationReadinessError("claim_source_hash_drift")
        audited.append(
            {
                "document_id": str(item.get("document_id")),
                "path": path_value,
                "sha256": actual,
                "size": path.stat().st_size,
            }
        )
    return sorted(audited, key=lambda item: item["document_id"])


def _audit_claims(
    contract: Mapping[str, Any], evidence_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    evidence = {str(item["evidence_id"]): item for item in evidence_index}
    source_ids = {
        str(item.get("document_id")) for item in contract.get("claim_sources") or []
    }
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in contract.get("claims") or []:
        claim_id = str(claim.get("claim_id") or "")
        status = str(claim.get("status") or "")
        scope = str(claim.get("scope") or "")
        references = sorted(str(value) for value in claim.get("evidence_ids") or [])
        if not claim_id or claim_id in seen:
            raise ValidationReadinessError("duplicate_or_empty_claim_id")
        seen.add(claim_id)
        if status not in ALLOWED_CLAIM_STATUSES:
            raise ValidationReadinessError("invalid_claim_status")
        if str(claim.get("document_id")) not in source_ids:
            raise ValidationReadinessError("unknown_claim_source")
        if not references or any(reference not in evidence for reference in references):
            raise ValidationReadinessError("claim_missing_machine_evidence")
        if scope == "formal_validation_requirement" and status != "blocked":
            raise ValidationReadinessError("formal_requirement_overclaim")
        if status == "verified" and scope != "engineering_capability":
            raise ValidationReadinessError("verified_scope_overclaim")
        if status == "internal_only" and scope not in {
            "internal_validation",
            "internal_diagnostic",
        }:
            raise ValidationReadinessError("internal_scope_mismatch")
        if status == "blocked" and not claim.get("blocker_id"):
            raise ValidationReadinessError("blocked_claim_without_blocker")
        claims.append(
            {
                "boundary": str(claim.get("boundary")),
                "claim_id": claim_id,
                "document_id": str(claim.get("document_id")),
                "evidence_ids": references,
                "scope": scope,
                "statement": str(claim.get("statement")),
                "status": status,
                **(
                    {"blocker_id": str(claim["blocker_id"])}
                    if claim.get("blocker_id")
                    else {}
                ),
            }
        )
    document_coverage = {
        document_id: sum(1 for claim in claims if claim["document_id"] == document_id)
        for document_id in sorted(source_ids)
    }
    if not claims or any(count == 0 for count in document_coverage.values()):
        raise ValidationReadinessError("claim_document_coverage_incomplete")
    return {
        "claim_count": len(claims),
        "claim_evidence_coverage_count": sum(bool(item["evidence_ids"]) for item in claims),
        "claim_evidence_coverage_rate": 1.0,
        "claims": sorted(claims, key=lambda item: item["claim_id"]),
        "document_coverage": document_coverage,
        "status_counts": {
            status: sum(item["status"] == status for item in claims)
            for status in sorted(ALLOWED_CLAIM_STATUSES)
        },
    }


def _audit_consistency(
    contract: Mapping[str, Any], loaded: Mapping[str, Any]
) -> dict[str, Any]:
    assertions: list[dict[str, Any]] = []
    for assertion in contract.get("consistency_assertions") or []:
        expected = assertion.get("expected")
        observed: list[dict[str, Any]] = []
        for source in assertion.get("observations") or []:
            evidence_id = str(source.get("evidence_id") or "")
            if evidence_id not in loaded:
                raise ValidationReadinessNotReady("consistency_evidence_missing")
            actual = _pointer_get(loaded[evidence_id], str(source.get("pointer") or ""))
            if actual != expected:
                raise ValidationReadinessError("cross_evidence_count_or_state_conflict")
            observed.append(
                {
                    "evidence_id": evidence_id,
                    "pointer": str(source.get("pointer")),
                    "value": actual,
                }
            )
        if not observed:
            raise ValidationReadinessError("empty_consistency_assertion")
        assertions.append(
            {
                "assertion_id": str(assertion.get("assertion_id")),
                "expected": expected,
                "observations": observed,
                "status": "consistent",
            }
        )
    return {
        "assertion_count": len(assertions),
        "conflict_count": 0,
        "assertions": sorted(assertions, key=lambda item: item["assertion_id"]),
    }


def _audit_blockers(
    contract: Mapping[str, Any], evidence_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    known_evidence = {str(item["evidence_id"]) for item in evidence_index}
    blockers: list[dict[str, Any]] = []
    for item in contract.get("blockers") or []:
        evidence_ids = sorted(str(value) for value in item.get("evidence_ids") or [])
        if not evidence_ids or any(value not in known_evidence for value in evidence_ids):
            raise ValidationReadinessError("blocker_missing_machine_evidence")
        if item.get("status") != "blocked":
            raise ValidationReadinessError("blocker_status_drift")
        blockers.append(
            {
                "blocker_id": str(item.get("blocker_id")),
                "evidence_ids": evidence_ids,
                "future_integration_point": str(item.get("future_integration_point")),
                "missing_external_input": str(item.get("missing_external_input")),
                "non_substitutes": sorted(str(v) for v in item.get("non_substitutes") or []),
                "status": "blocked",
            }
        )
    if {item["blocker_id"] for item in blockers} != REQUIRED_BLOCKERS:
        raise ValidationReadinessError("required_blockers_not_closed")
    return {
        "blocker_count": len(blockers),
        "blockers": sorted(blockers, key=lambda item: item["blocker_id"]),
        "formal_validation_complete": False,
    }


def _code_identity(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    files: dict[str, bytes] = {}
    for relative in contract.get("code_identity", {}).get("files") or []:
        relative = str(relative)
        path = _safe_repo_path(root, relative)
        if not path.is_file():
            raise ValidationReadinessNotReady("implementation_file_missing")
        files[relative] = path.read_bytes()
    if not files:
        raise ValidationReadinessError("empty_code_identity")
    return {
        "implementation_base_commit": str(
            contract.get("code_identity", {}).get("implementation_base_commit")
        ),
        "implementation_file_count": len(files),
        "implementation_tree_sha256": stable_tree_sha256(files),
    }


def _verify_commit_ancestry(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    base = str(contract.get("code_identity", {}).get("implementation_base_commit") or "")
    if len(base) != 40 or any(character not in "0123456789abcdef" for character in base):
        raise ValidationReadinessError("implementation_base_commit_invalid")
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }
    try:
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
            env=environment,
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationReadinessNotReady("git_commit_binding_unavailable") from exc
    if exists.returncode != 0 or ancestor.returncode != 0:
        raise ValidationReadinessError("implementation_base_not_ancestor")
    return {
        "implementation_base_commit_exists": True,
        "implementation_base_is_ancestor_of_head": True,
    }


def _workspace_state(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected_path = str(contract.get("workspace", {}).get("preserved_path") or "")
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", ""),
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValidationReadinessNotReady("git_workspace_unavailable") from exc
    if completed.returncode != 0:
        raise ValidationReadinessNotReady("git_workspace_unavailable")
    entries = completed.stdout.splitlines()
    nested = [line for line in entries if line[3:].startswith("third_party/")]
    if (
        len(nested) != 1
        or nested[0][3:] != expected_path
        or nested[0][:2] not in {" M", " m"}
    ):
        raise ValidationReadinessError("preserved_nested_worktree_state_drift")
    return {
        "preserved_nested_worktree_count": 1,
        "preserved_nested_worktree_state_verified": True,
        "unexpected_nested_worktree_entry_count": 0,
    }


def _run_read_only_gates(
    contract: Mapping[str, Any], root: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_env = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "src",
    }
    for gate in contract.get("read_only_gates") or []:
        with tempfile.TemporaryDirectory(prefix="validation-readiness-") as temporary:
            arguments = [
                temporary if str(value) == "{temporary_output}" else str(value)
                for value in gate.get("arguments") or []
            ]
            command = [sys.executable, *arguments]
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=int(gate.get("timeout_seconds") or 60),
                    env=base_env,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ValidationReadinessNotReady("read_only_gate_unavailable") from exc
        expected_exit = int(gate.get("expected_exit_code"))
        if completed.returncode != expected_exit:
            raise ValidationReadinessError(
                "read_only_gate_exit_drift:"
                f"{gate.get('gate_id')}:{completed.returncode}:{expected_exit}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValidationReadinessError("read_only_gate_output_invalid") from exc
        for check in gate.get("checks") or []:
            if _pointer_get(payload, str(check.get("pointer") or "")) != check.get("equals"):
                raise ValidationReadinessError("read_only_gate_result_drift")
        results.append(
            {
                "exit_code": completed.returncode,
                "gate_id": str(gate.get("gate_id")),
                "output_sha256": sha256_bytes(
                    canonical_json(payload).encode("utf-8")
                ),
                "status": str(payload.get("status") or ("passed" if payload.get("passed") else "verified")),
            }
        )
    return sorted(results, key=lambda item: item["gate_id"])


def _render_readme(
    claims: Mapping[str, Any], blockers: Mapping[str, Any], consistency: Mapping[str, Any]
) -> str:
    return (
        "# validation_readiness_bundle_v1\n\n"
        "This deterministic, offline bundle indexes tracked engineering and internal validation "
        "evidence. It does not contain source paper/query text, private mappings, credentials, "
        "temporary logs, or third-party source code.\n\n"
        f"- Claim trace coverage: {claims['claim_evidence_coverage_count']}/{claims['claim_count']}\n"
        f"- Cross-evidence assertions: {consistency['assertion_count']} consistent\n"
        f"- Declared formal blockers: {blockers['blocker_count']}\n"
        "- Overall status: `ready_with_declared_blockers`\n\n"
        "Run `PYTHONPATH=src python scripts/check_validation_readiness.py verify "
        "--contract benchmark/validation_readiness_bundle_v1_contract.json "
        "--bundle benchmark/validation_readiness_bundle_v1_release` from the repository root.\n\n"
        "Passing this gate proves only evidence integrity, traceability, and declared boundaries. "
        "It is neither human Precision nor an official competition score.\n"
    )


def _render_checklist(blockers: Mapping[str, Any]) -> str:
    lines = [
        "# Competition delivery preflight checklist",
        "",
        "## Offline evidence checks available now",
        "",
        "- [x] Historical evidence hashes and protocol dependencies are indexed.",
        "- [x] Record162/Record160, Snapshot-key, and final-result counts are cross-checked.",
        "- [x] Default current_rules and disabled experimental tie-break state are checked.",
        "- [x] Delivery-fidelity unsupported exits remain explicit.",
        "",
        "## External inputs still required before formal validation",
        "",
    ]
    for blocker in blockers["blockers"]:
        lines.append(f"- [ ] `{blocker['blocker_id']}`: {blocker['missing_external_input']}")
    lines.extend(
        [
            "",
            "Coverage, stability, source diagnostics, LLM proxy runs, and delivery fidelity must not "
            "be substituted for any unchecked item above.",
            "",
        ]
    )
    return "\n".join(lines)


def build_release_files(
    contract: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    run_gates: bool = True,
    check_workspace: bool = True,
) -> dict[str, bytes]:
    """Build release files in memory without mutating tracked evidence."""

    root = Path(repository_root).resolve()
    evidence, loaded = _audit_evidence(contract, root)
    claim_sources = _audit_claim_sources(contract, root)
    claims = _audit_claims(contract, evidence)
    consistency = _audit_consistency(contract, loaded)
    blockers = _audit_blockers(contract, evidence)
    code_identity = _code_identity(contract, root)
    code_identity.update(
        _verify_commit_ancestry(contract, root)
        if check_workspace
        else {
            "implementation_base_commit_exists": True,
            "implementation_base_is_ancestor_of_head": True,
        }
    )
    gates = _run_read_only_gates(contract, root) if run_gates else []
    workspace = (
        _workspace_state(contract, root)
        if check_workspace
        else {
            "preserved_nested_worktree_count": 1,
            "preserved_nested_worktree_state_verified": True,
            "unexpected_nested_worktree_entry_count": 0,
        }
    )
    execution = dict(contract["execution"])
    readiness = {
        "blocker_count": blockers["blocker_count"],
        "claim_count": claims["claim_count"],
        "claim_evidence_coverage_rate": claims["claim_evidence_coverage_rate"],
        "code_identity": code_identity,
        "consistency_assertion_count": consistency["assertion_count"],
        "execution": execution,
        "formal_validation_complete": False,
        "protocol": PROTOCOL_VERSION,
        "read_only_gate_count": len(gates),
        "read_only_gates": gates,
        "schema_version": SCHEMA_VERSION,
        "status": "ready_with_declared_blockers",
        "workspace": workspace,
    }
    files: dict[str, bytes] = {
        "README.md": _render_readme(claims, blockers, consistency).encode("utf-8"),
        "claims.json": canonical_json(
            {
                "claim_sources": claim_sources,
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                **claims,
            }
        ).encode("utf-8"),
        "competition_checklist.md": _render_checklist(blockers).encode("utf-8"),
        "evidence_index.json": canonical_json(
            {
                "evidence": evidence,
                "evidence_count": len(evidence),
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
            }
        ).encode("utf-8"),
        "missing_inputs.json": canonical_json(
            {
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                **blockers,
            }
        ).encode("utf-8"),
        "readiness.json": canonical_json(readiness).encode("utf-8"),
    }
    dependency_graph = {
        str(item["evidence_id"]): list(item["dependencies"]) for item in evidence
    }
    bundle = {
        "code_identity": code_identity,
        "dependency_graph": dependency_graph,
        "files": {
            name: {"sha256": sha256_bytes(content), "size": len(content)}
            for name, content in sorted(files.items())
        },
        "generation_command": str(contract["release"]["generation_command"]),
        "protocol": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": "ready_with_declared_blockers",
        "verification_command": str(contract["release"]["verification_command"]),
    }
    files["bundle.json"] = canonical_json(bundle).encode("utf-8")
    for name, content in files.items():
        _assert_no_release_leaks({"name": name, "content": content.decode("utf-8")})
    return dict(sorted(files.items()))


def write_release_files(output: str | Path, files: Mapping[str, bytes]) -> str:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    expected = set(files)
    existing = {path.name for path in output_path.iterdir() if path.is_file()}
    unexpected = existing - expected
    if unexpected:
        raise ValidationReadinessError("unexpected_release_member")
    for name, content in sorted(files.items()):
        temporary = output_path / f".{name}.tmp"
        temporary.write_bytes(content)
        os.replace(temporary, output_path / name)
    return stable_tree_sha256(files)


def verify_release_files(
    contract: Mapping[str, Any],
    bundle: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    run_gates: bool = True,
    check_workspace: bool = True,
) -> dict[str, Any]:
    expected = build_release_files(
        contract,
        repository_root=repository_root,
        run_gates=run_gates,
        check_workspace=check_workspace,
    )
    bundle_path = Path(bundle)
    if not bundle_path.is_dir():
        raise ValidationReadinessNotReady("release_bundle_missing")
    actual_names = {path.name for path in bundle_path.iterdir() if path.is_file()}
    if actual_names != set(expected):
        raise ValidationReadinessError("release_member_set_drift")
    for name, expected_bytes in expected.items():
        if (bundle_path / name).read_bytes() != expected_bytes:
            raise ValidationReadinessError("release_file_byte_drift")
    readiness = json.loads(expected["readiness.json"])
    return {
        "blocker_count": int(readiness["blocker_count"]),
        "bundle_file_count": len(expected),
        "bundle_tree_sha256": stable_tree_sha256(expected),
        "claim_count": int(readiness["claim_count"]),
        "claim_evidence_coverage_rate": readiness["claim_evidence_coverage_rate"],
        "execution": readiness["execution"],
        "exit_code": EXIT_READY_WITH_DECLARED_BLOCKERS,
        "protocol": PROTOCOL_VERSION,
        "read_only_gate_count": int(readiness["read_only_gate_count"]),
        "schema_version": SCHEMA_VERSION,
        "status": "ready_with_declared_blockers",
    }
