"""Deterministic Python dependency locking and offline-install qualification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from collections import defaultdict, deque
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from scholar_agent.evaluation.release_candidate_reproducibility import (
    EXECUTION,
    DIST_INFO,
    ReleaseCandidateError,
    build_wheel,
    canonical_json,
    freeze_contract,
    materialize_source,
    sha256_bytes,
    sha256_file,
    stable_digest,
)


PROTOCOL = "python_dependency_lock_v1"
SCHEMA_VERSION = "1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4


class DependencyLockError(RuntimeError):
    pass


class DependencyLockNotReady(DependencyLockError):
    pass


def _environment() -> dict[str, str]:
    values = default_environment()
    return {
        "implementation_name": values["implementation_name"],
        "platform_machine": values["platform_machine"],
        "python_full_version": values["python_full_version"],
        "python_version": values["python_version"],
        "sys_platform": values["sys_platform"],
    }


def _read_direct(path: Path, group: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        if name in seen:
            raise DependencyLockError("duplicate_direct_dependency")
        seen.add(name)
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise DependencyLockError("direct_dependency_not_exact")
        rows.append(
            {
                "declaration": value,
                "group": group,
                "line": line_number,
                "marker": str(requirement.marker) if requirement.marker else None,
                "name": name,
                "version": specifiers[0].version,
            }
        )
    if not rows:
        raise DependencyLockError("empty_dependency_group")
    return sorted(rows, key=lambda item: item["name"])


def _metadata_sha256(distribution: metadata.Distribution) -> str:
    content = distribution.read_text("METADATA")
    if content is None:
        raise DependencyLockNotReady("distribution_metadata_missing")
    return sha256_bytes(content.encode("utf-8"))


def _license(distribution: metadata.Distribution) -> str:
    value = (
        distribution.metadata.get("License-Expression")
        or distribution.metadata.get("License")
        or "unknown"
    )
    return value.strip() if value.strip() and value.strip().upper() != "UNKNOWN" else "unknown"


def _cycles(graph: Mapping[str, set[str]]) -> list[list[str]]:
    found: set[tuple[str, ...]] = set()

    def visit(node: str, stack: list[str], active: set[str]) -> None:
        if node in active:
            start = stack.index(node)
            cycle = stack[start:] + [node]
            rotations = [tuple(cycle[index:-1] + cycle[:index] + [cycle[index]]) for index in range(len(cycle) - 1)]
            found.add(min(rotations))
            return
        active.add(node)
        stack.append(node)
        for child in sorted(graph.get(node, set())):
            visit(child, stack, active)
        stack.pop()
        active.remove(node)

    for node in sorted(graph):
        visit(node, [], set())
    return [list(value) for value in sorted(found)]


def _pip_cache_wheels() -> list[Path]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "cache", "dir"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={"LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")},
    )
    if completed.returncode != 0:
        return []
    root = Path(completed.stdout.strip())
    return sorted(path for path in root.rglob("*.whl") if path.is_file()) if root.is_dir() else []


def _artifact_inventory(packages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = {(item["name"], item["version"]) for item in packages}
    compatible_tags = set(sys_tags())
    matched: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in _pip_cache_wheels():
        try:
            name, version, _build, tags = parse_wheel_filename(path.name)
        except ValueError:
            continue
        key = (canonicalize_name(str(name)), str(version))
        if key in required and compatible_tags.intersection(tags):
            matched[key].append(
                {"filename": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
            )
    artifacts = []
    missing = []
    for name, version in sorted(required):
        values = sorted(matched.get((name, version), []), key=lambda item: (item["filename"], item["sha256"]))
        if not values:
            missing.append({"name": name, "version": version})
        artifacts.extend({"name": name, "version": version, **item} for item in values)
    return {
        "qualified": not missing,
        "artifacts": artifacts,
        "missing": missing,
        "artifact_count": len(artifacts),
        "required_package_count": len(required),
    }


def build_manifest(repository_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("protocol") != PROTOCOL or protocol.get("schema_version") != SCHEMA_VERSION:
        raise DependencyLockError("protocol_version_mismatch")
    if protocol.get("environment") != _environment():
        raise DependencyLockNotReady("environment_identity_mismatch")
    declarations: list[dict[str, Any]] = []
    for group in ("runtime", "development"):
        declarations.extend(
            _read_direct(repository_root / protocol["authoritative_declarations"][group], group)
        )
    names_by_group: dict[str, set[str]] = defaultdict(set)
    for item in declarations:
        names_by_group[item["group"]].add(item["name"])
    overlap = names_by_group["runtime"] & names_by_group["development"]
    if overlap:
        raise DependencyLockError("direct_dependency_group_overlap")

    environment = default_environment()
    queue: deque[tuple[str, Requirement, str | None]] = deque()
    for item in declarations:
        queue.append((item["group"], Requirement(item["declaration"]), None))
    package_groups: dict[str, set[str]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    constraints: dict[str, set[str]] = defaultdict(set)
    excluded_markers: list[dict[str, str]] = []
    distributions: dict[str, metadata.Distribution] = {}
    visited_groups: set[tuple[str, str]] = set()
    while queue:
        group, requirement, parent = queue.popleft()
        name = canonicalize_name(requirement.name)
        if requirement.marker and not requirement.marker.evaluate(environment):
            excluded_markers.append(
                {"group": group, "parent": parent or "direct", "requirement": str(requirement)}
            )
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise DependencyLockNotReady(f"installed_distribution_missing:{name}") from exc
        if requirement.specifier and not requirement.specifier.contains(distribution.version, prereleases=True):
            raise DependencyLockError(f"installed_version_conflict:{name}")
        package_groups[name].add(group)
        constraints[name].add(str(requirement))
        if parent is not None:
            graph[parent].add(name)
        distributions[name] = distribution
        visit = (group, name)
        if visit in visited_groups:
            continue
        visited_groups.add(visit)
        for raw in distribution.requires or []:
            child = Requirement(raw)
            if child.marker and not child.marker.evaluate(environment):
                excluded_markers.append(
                    {"group": group, "parent": name, "requirement": str(child)}
                )
                continue
            queue.append((group, child, name))

    cycles = _cycles(graph)
    if cycles:
        raise DependencyLockError("dependency_cycle_detected")
    packages = []
    direct_lookup = {(item["group"], item["name"]): item for item in declarations}
    for name in sorted(distributions):
        distribution = distributions[name]
        groups = sorted(package_groups[name])
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "groups": groups,
                "direct_groups": [group for group in groups if (group, name) in direct_lookup],
                "dependencies": sorted(graph.get(name, set())),
                "incoming_requirements": sorted(constraints[name]),
                "source_evidence": {
                    "kind": "installed_distribution_metadata",
                    "metadata_sha256": _metadata_sha256(distribution),
                },
                "license": _license(distribution),
            }
        )
    artifacts = _artifact_inventory(packages)
    runtime_direct = [item["declaration"] for item in declarations if item["group"] == "runtime"]
    development_direct = [item["declaration"] for item in declarations if item["group"] == "development"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": protocol["release_source_commit"],
        "environment": _environment(),
        "declarations": sorted(declarations, key=lambda item: (item["group"], item["name"])),
        "runtime_requires_dist": runtime_direct,
        "development_direct": development_direct,
        "packages": packages,
        "package_count": len(packages),
        "runtime_package_count": sum("runtime" in item["groups"] for item in packages),
        "development_only_package_count": sum(item["groups"] == ["development"] for item in packages),
        "excluded_markers": sorted(excluded_markers, key=lambda item: (item["group"], item["parent"], item["requirement"])),
        "dependency_cycles": cycles,
        "offline_artifacts": artifacts,
        "lock_qualified": True,
        "offline_install_qualified": artifacts["qualified"],
        "status": "dependency_lock_qualified" if artifacts["qualified"] else "not_ready_missing_verified_version_or_artifact",
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def lock_text(manifest: Mapping[str, Any], group: str) -> bytes:
    if group not in {"runtime", "development"}:
        raise DependencyLockError("unknown_dependency_group")
    packages = [item for item in manifest["packages"] if group in item["groups"]]
    lines = [
        f"# {PROTOCOL}",
        f"# group: {group}",
        f"# environment: python {manifest['environment']['python_full_version']} {manifest['environment']['sys_platform']} {manifest['environment']['platform_machine']}",
        *[f"{item['name']}=={item['version']}" for item in sorted(packages, key=lambda item: item["name"])],
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def verify_manifest(
    repository_root: Path,
    protocol: Mapping[str, Any],
    tracked_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    actual = build_manifest(repository_root, protocol)
    violations = []
    if actual != tracked_manifest:
        violations.append("manifest_drift")
    for group in ("runtime", "development"):
        output = repository_root / protocol["lock_outputs"][group]
        expected = lock_text(actual, group)
        if not output.is_file() or output.read_bytes() != expected:
            violations.append(f"{group}_lock_drift")
    runtime_names = {item["name"] for item in actual["packages"] if "runtime" in item["groups"]}
    if {"pytest", "httpx"} & runtime_names:
        violations.append("development_dependency_in_runtime_closure")
    status = "lock_or_metadata_violation" if violations else actual["status"]
    exit_code = EXIT_VIOLATION if violations else (
        EXIT_QUALIFIED if actual["offline_install_qualified"] else EXIT_NOT_READY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "violations": violations,
        "lock_qualified": not violations,
        "offline_install_qualified": actual["offline_install_qualified"],
        "missing_artifacts": actual["offline_artifacts"]["missing"],
        "package_count": actual["package_count"],
        "runtime_package_count": actual["runtime_package_count"],
        "development_only_package_count": actual["development_only_package_count"],
        "manifest_sha256": stable_digest(actual),
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def wheel_requires_dist(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        content = archive.read(f"{DIST_INFO}/METADATA").decode("utf-8")
    return sorted(line.removeprefix("Requires-Dist: ") for line in content.splitlines() if line.startswith("Requires-Dist: "))


def verify_wheel_metadata(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    actual = wheel_requires_dist(path)
    expected = sorted(manifest["runtime_requires_dist"])
    development = {item["name"] for item in manifest["declarations"] if item["group"] == "development"}
    leaked = sorted(
        value for value in actual if canonicalize_name(Requirement(value).name) in development
    )
    violations = []
    if actual != expected:
        violations.append("wheel_requires_dist_mismatch")
    if leaked:
        violations.append("development_dependency_in_wheel_metadata")
    return {
        "passed": not violations,
        "violations": violations,
        "requires_dist": actual,
        "development_leaks": leaked,
    }


def freeze_release_contract(
    repository_root: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    release_spec = json.loads(
        (repository_root / "benchmark/release_candidate_reproducibility_v1_spec.json").read_text(encoding="utf-8")
    )
    release_spec["source_commit"] = protocol["release_source_commit"]
    contract, locks = freeze_contract(repository_root, release_spec)
    frontend_protocol = json.loads(
        (repository_root / "benchmark/frontend_reproducible_build_v1_protocol.json").read_text(encoding="utf-8")
    )
    contract["frontend_canonical_staging"] = frontend_protocol["canonical_staging"]
    contract["python_lock"] = {
        "path": "benchmark/python_dependency_lock_v1_release_closure.json",
        "sha256": stable_digest(locks["python"]),
    }
    contract["python_dependency_lock"] = {
        "protocol": PROTOCOL,
        "manifest_sha256": stable_digest(manifest),
        "runtime_requires_dist": manifest["runtime_requires_dist"],
        "lock_qualified": manifest["lock_qualified"],
        "offline_install_qualified": manifest["offline_install_qualified"],
    }
    return contract, locks["python"]


def offline_install(
    repository_root: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    missing = manifest["offline_artifacts"]["missing"]
    if missing:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "not_ready_missing_verified_version_or_artifact",
            "exit_code": EXIT_NOT_READY,
            "missing_artifacts": missing,
            "venv_results": [],
            "execution": EXECUTION,
            "formal_validation_complete": False,
        }
    artifacts_by_filename = {
        path.name: path
        for path in _pip_cache_wheels()
        if path.name in {item["filename"] for item in manifest["offline_artifacts"]["artifacts"]}
    }
    with tempfile.TemporaryDirectory(prefix="python-lock-offline-") as temporary:
        temp = Path(temporary)
        source = temp / "source"
        materialize_source(repository_root, contract, source)
        application_wheel = temp / "wheelhouse/spar_scholar_agent-0.1.0-py3-none-any.whl"
        application_wheel.parent.mkdir()
        build_wheel(source, application_wheel, contract)
        for name, path in artifacts_by_filename.items():
            shutil.copyfile(path, application_wheel.parent / name)
        results = []
        for index in range(2):
            profile = temp / f"profile-{index}"
            home, tmpdir, environment = profile / "home", profile / "tmp", profile / "environment"
            home.mkdir(parents=True); tmpdir.mkdir(); venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / "bin/python"
            env = {
                "HOME": str(home), "TMPDIR": str(tmpdir), "LANG": "C", "LC_ALL": "C",
                "PATH": str(environment / "bin"), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
            install = subprocess.run(
                [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(application_wheel.parent), "-r", str(repository_root / "requirements-runtime.lock"), str(application_wheel)],
                cwd=profile, env=env, capture_output=True, text=True, timeout=180, check=False,
            )
            smoke = subprocess.run(
                [str(python), "-c", "from scholar_agent.app.main import app; import rank_bm25; assert len(app.routes) > 0"],
                cwd=profile, env=env, capture_output=True, text=True, timeout=30, check=False,
            ) if install.returncode == 0 else None
            help_result = subprocess.run(
                [str(python), "-m", "uvicorn", "--help"], cwd=profile, env=env,
                capture_output=True, text=True, timeout=30, check=False,
            ) if install.returncode == 0 else None
            uninstall = subprocess.run(
                [str(python), "-m", "pip", "uninstall", "-y", "spar-scholar-agent"], cwd=profile, env=env,
                capture_output=True, text=True, timeout=30, check=False,
            ) if install.returncode == 0 else None
            residue = subprocess.run(
                [str(python), "-c", "import importlib.util; raise SystemExit(1 if importlib.util.find_spec('scholar_agent') else 0)"],
                cwd=profile, env=env, capture_output=True, text=True, timeout=30, check=False,
            ) if uninstall and uninstall.returncode == 0 else None
            passed = all(value is not None and value.returncode == 0 for value in (install, smoke, help_result, uninstall, residue))
            results.append({"profile": index, "passed": passed})
        qualified = all(item["passed"] for item in results)
        return {
            "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
            "status": "dependency_lock_qualified" if qualified else "lock_or_metadata_violation",
            "exit_code": EXIT_QUALIFIED if qualified else EXIT_VIOLATION,
            "missing_artifacts": [], "venv_results": results,
            "execution": EXECUTION, "formal_validation_complete": False,
        }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
