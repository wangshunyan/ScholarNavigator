"""Offline qualification for path-independent Next/webpack release builds."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Mapping

from scholar_agent.evaluation.release_candidate_reproducibility import (
    EXECUTION,
    ReleaseCandidateError,
    ReleaseCandidateNotReady,
    canonical_json,
    double_build,
    freeze_contract,
    sha256_bytes,
    sha256_file,
    stable_digest,
)


PROTOCOL = "frontend_reproducible_build_v1"
SCHEMA_VERSION = "1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_UPSTREAM = 3
EXIT_USAGE = 4


def freeze_release_contract(
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a new release contract without mutating the historical v1 files."""

    if protocol.get("protocol") != PROTOCOL or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseCandidateError("frontend_protocol_version_mismatch")
    release_spec = json.loads(
        (repository_root / "benchmark/release_candidate_reproducibility_v1_spec.json").read_text(
            encoding="utf-8"
        )
    )
    release_spec["source_commit"] = protocol["release_source_commit"]
    contract, locks = freeze_contract(repository_root, release_spec)
    contract["frontend_canonical_staging"] = dict(protocol["canonical_staging"])
    contract["frontend_reproducible_build_protocol_sha256"] = stable_digest(protocol)
    return contract, locks


def _archive_members(path: Path) -> dict[str, bytes]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members: dict[str, bytes] = {}
            for item in archive.getmembers():
                if not item.isfile() or item.name in members:
                    continue
                extracted = archive.extractfile(item)
                if extracted is None:
                    raise ReleaseCandidateError("frontend_archive_member_unreadable")
                members[item.name] = extracted.read()
            return members
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseCandidateError("frontend_archive_invalid") from exc


def _member_hashes(members: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": path, "size": len(content), "sha256": sha256_bytes(content)}
        for path, content in sorted(members.items())
    ]


def diagnose_archive_pair(first: Path, second: Path) -> dict[str, Any]:
    """Report literal byte differences; no normalized comparison grants a pass."""

    left = _archive_members(first)
    right = _archive_members(second)
    all_paths = sorted(set(left) | set(right))
    different = [path for path in all_paths if left.get(path) != right.get(path)]
    same_path_differences = [path for path in different if path in left and path in right]
    chunk_members = lambda values: {path for path in values if path.startswith("frontend/static/chunks/")}
    build_id_left = left.get("frontend/BUILD_ID")
    build_id_right = right.get("frontend/BUILD_ID")
    forbidden_patterns = (b"/Users/", b"/home/", b"sourceMappingURL")
    return {
        "archive_sha256_equal": sha256_file(first) == sha256_file(second),
        "build_id_equal": build_id_left is not None and build_id_left == build_id_right,
        "differing_member_count": len(different),
        "direct_same_path_differences": same_path_differences,
        "member_count": {"first": len(left), "second": len(right)},
        "member_sets_equal": set(left) == set(right),
        "chunk_member_symmetric_difference_count": len(chunk_members(left) ^ chunk_members(right)),
        "trace_difference_count": sum(path.endswith(".nft.json") for path in different),
        "source_map_member_count": sum(path.endswith(".map") for path in all_paths),
        "forbidden_literal_counts": {
            pattern.decode(): sum(content.count(pattern) for content in left.values())
            + sum(content.count(pattern) for content in right.values())
            for pattern in forbidden_patterns
        },
    }


def verify_runtime_archive(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    members = _archive_members(path)
    violations: list[dict[str, str]] = []
    required = {
        "frontend/BUILD_ID",
        "frontend/build-manifest.json",
        "frontend/routes-manifest.json",
        "frontend/server/app/index.html",
        "frontend/server/app/index.rsc",
    }
    for missing in sorted(required - set(members)):
        violations.append({"invariant": "required_runtime_member", "path": missing})
    expected_build_id = f"spar-release-{str(contract['source_commit'])[:20]}".encode()
    if members.get("frontend/BUILD_ID") != expected_build_id:
        violations.append({"invariant": "stable_build_id", "path": "frontend/BUILD_ID"})
    try:
        routes = json.loads(members.get("frontend/routes-manifest.json", b"{}").decode())
        build_manifest = json.loads(members.get("frontend/build-manifest.json", b"{}").decode())
    except (UnicodeError, json.JSONDecodeError):
        violations.append({"invariant": "manifest_json", "path": "frontend"})
        routes, build_manifest = {}, {}
    if not any(route.get("page") == "/" for route in routes.get("staticRoutes") or []):
        violations.append({"invariant": "root_route", "path": "frontend/routes-manifest.json"})
    declared_assets: set[str] = set()
    for values in build_manifest.values():
        if isinstance(values, list):
            declared_assets.update(value for value in values if isinstance(value, str) and value.startswith("static/"))
    for asset in sorted(declared_assets):
        if f"frontend/{asset}" not in members:
            violations.append({"invariant": "manifest_static_reference", "path": asset})
    html = members.get("frontend/server/app/index.html", b"")
    for raw in re.findall(rb'/_next/static/([^"\'?\\]+)', html):
        target = "frontend/static/" + raw.decode("utf-8", "strict")
        if target not in members:
            violations.append({"invariant": "html_static_reference", "path": target})
    if b"self.__next_f.push" not in html:
        violations.append({"invariant": "hydration_payload", "path": "frontend/server/app/index.html"})
    api_type = next(
        (item for item in contract["source_manifest"] if item["path"] == "frontend/src/types/api.ts"),
        None,
    )
    if not api_type:
        violations.append({"invariant": "api_type_contract", "path": "frontend/src/types/api.ts"})
    return {
        "passed": not violations,
        "violations": violations,
        "route_count": len(routes.get("staticRoutes") or []),
        "declared_static_asset_count": len(declared_assets),
        "hydration_payload_present": b"self.__next_f.push" in html,
        "api_type_contract_sha256": api_type["sha256"] if api_type else None,
        "member_count": len(members),
        "member_tree_sha256": stable_digest(_member_hashes(members)),
        "members": _member_hashes(members),
    }


def run_qualification(
    repository_root: Path,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    legacy_contract = copy.deepcopy(contract)
    legacy_contract.pop("frontend_canonical_staging", None)
    legacy_report = double_build(repository_root, legacy_contract, output_root / "legacy")
    fixed_report = double_build(repository_root, contract, output_root / "fixed")
    legacy_paths = [
        output_root / "legacy/profile-a/outputs/frontend-static.tar.gz",
        output_root / "legacy/different-parent/profile-b/outputs/frontend-static.tar.gz",
    ]
    fixed_paths = [
        output_root / "fixed/profile-a/outputs/frontend-static.tar.gz",
        output_root / "fixed/different-parent/profile-b/outputs/frontend-static.tar.gz",
    ]
    legacy = diagnose_archive_pair(*legacy_paths)
    fixed = diagnose_archive_pair(*fixed_paths)
    runtime = [verify_runtime_archive(path, contract) for path in fixed_paths]
    qualified = fixed["archive_sha256_equal"] and all(item["passed"] for item in runtime)
    if qualified:
        status, exit_code = "qualified", EXIT_QUALIFIED
    elif not fixed["archive_sha256_equal"] and all(item["passed"] for item in runtime):
        status, exit_code = "not_qualified_upstream_limitation", EXIT_UPSTREAM
    else:
        status, exit_code = "reproducibility_or_runtime_violation", EXIT_VIOLATION
    historical_path = repository_root / "benchmark/release_candidate_reproducibility_v1_evidence/current.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "source_commit": contract["source_commit"],
        "protocol_sha256": stable_digest(protocol),
        "release_contract_sha256": stable_digest(contract),
        "root_cause": {
            "classification": "webpack_chunk_assignment_depends_on_source_parent_context",
            "evidence": legacy,
            "canonical_staging_control_passed": fixed["archive_sha256_equal"],
        },
        "fixed_pair": fixed,
        "runtime_fidelity": runtime,
        "frontend_archive": {
            "sha256": sha256_file(fixed_paths[0]) if qualified else None,
            "member_tree_sha256": runtime[0]["member_tree_sha256"] if qualified else None,
            "member_count": runtime[0]["member_count"] if qualified else None,
            "members": runtime[0]["members"] if qualified else [],
        },
        "release_candidate": {
            "qualified": qualified and not fixed_report["dependency_violations"],
            "frontend_qualified": qualified,
            "remaining_dependency_violations": fixed_report["dependency_violations"],
            "status": (
                "reproducible_release_ready"
                if qualified and not fixed_report["dependency_violations"]
                else "build_or_supply_chain_violation"
            ),
        },
        "historical_failure_evidence": {
            "path": "benchmark/release_candidate_reproducibility_v1_evidence/current.json",
            "sha256": sha256_file(historical_path),
        },
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def verify_evidence(
    evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    violations = []
    if evidence.get("protocol") != PROTOCOL:
        violations.append("protocol_mismatch")
    if evidence.get("protocol_sha256") != stable_digest(protocol):
        violations.append("protocol_digest_mismatch")
    if evidence.get("release_contract_sha256") != stable_digest(contract):
        violations.append("release_contract_digest_mismatch")
    if evidence.get("source_commit") != contract.get("source_commit"):
        violations.append("source_commit_mismatch")
    if evidence.get("status") != "qualified":
        violations.append("frontend_not_qualified")
    if not evidence.get("root_cause", {}).get("canonical_staging_control_passed"):
        violations.append("canonical_staging_control_failed")
    if any(not item.get("passed") for item in evidence.get("runtime_fidelity") or []):
        violations.append("runtime_fidelity_failed")
    fixed = evidence.get("fixed_pair") or {}
    if fixed.get("differing_member_count") != 0 or not fixed.get("archive_sha256_equal"):
        violations.append("fixed_pair_not_byte_identical")
    archive = evidence.get("frontend_archive") or {}
    members = archive.get("members") or []
    paths = [item.get("path") for item in members]
    if (
        archive.get("member_count") != len(members)
        or len(paths) != len(set(paths))
        or members != sorted(members, key=lambda item: item["path"])
        or archive.get("member_tree_sha256") != stable_digest(members)
    ):
        violations.append("frontend_member_manifest_invalid")
    runtime = evidence.get("runtime_fidelity") or []
    if len(runtime) != 2 or any(
        item.get("member_tree_sha256") != archive.get("member_tree_sha256")
        or item.get("member_count") != archive.get("member_count")
        or item.get("members") != members
        for item in runtime
    ):
        violations.append("runtime_member_manifest_mismatch")
    if not evidence.get("release_candidate", {}).get("frontend_qualified"):
        violations.append("release_frontend_qualification_missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "qualified" if not violations else "reproducibility_or_runtime_violation",
        "exit_code": EXIT_QUALIFIED if not violations else EXIT_VIOLATION,
        "violations": violations,
        "release_candidate_status": evidence.get("release_candidate", {}).get("status"),
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
