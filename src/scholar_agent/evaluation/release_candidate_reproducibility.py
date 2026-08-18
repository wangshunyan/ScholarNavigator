"""Deterministic, offline software release candidate builder and gate.

The release is source and engineering material only.  It deliberately excludes
benchmark run records, quality metrics, official submissions, untracked files,
``third_party`` and environment configuration.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.message import Message
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROTOCOL = "release_candidate_reproducibility_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
WHEEL_NAME = "spar_scholar_agent-0.1.0-py3-none-any.whl"
DIST_INFO = "spar_scholar_agent-0.1.0.dist-info"
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}


class ReleaseCandidateError(RuntimeError):
    """A release or supply-chain invariant was violated."""


class ReleaseCandidateNotReady(ReleaseCandidateError):
    """A declared offline dependency or source input is unavailable."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseCandidateError("unsafe_relative_path")
    if path.parts[0] == "third_party" or path.name == ".env":
        raise ReleaseCandidateError("prohibited_release_path")
    return path.as_posix()


def _git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=not binary,
        timeout=60,
        env={"LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")},
    )
    if completed.returncode != 0:
        raise ReleaseCandidateNotReady("git_input_unavailable")
    return completed.stdout


def source_manifest_from_git(root: Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    commit = str(spec["source_commit"])
    names = str(_git(root, "ls-tree", "-r", "--name-only", commit)).splitlines()
    selection = spec["source_selection"]
    exact = set(selection["exact"])
    prefixes = tuple(selection["prefixes"])
    suffixes = tuple(selection["suffixes"])
    selected = sorted(
        name for name in names
        if name in exact or name.startswith(prefixes) or name.endswith(suffixes)
    )
    if not selected or any(name.startswith("third_party/") or name == ".env" for name in selected):
        raise ReleaseCandidateError("source_selection_invalid")
    rows = []
    for name in selected:
        relative = _safe_relative(name)
        blob = _git(root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(blob, bytes)
        rows.append({"path": relative, "size": len(blob), "sha256": sha256_bytes(blob)})
    return rows


def build_python_lock(requirements_path: Path) -> dict[str, Any]:
    direct_lines = [
        line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    pending: list[tuple[str, str]] = []
    direct_names: set[str] = set()
    unpinned_direct_requirements = []
    for line in direct_lines:
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            unpinned_direct_requirements.append(line)
        name = canonicalize_name(requirement.name)
        direct_names.add(name)
        pending.append((name, line))
    resolved: dict[str, dict[str, Any]] = {}
    missing: set[str] = set()
    while pending:
        name, declaration = pending.pop(0)
        if name in resolved or name in missing:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.add(name)
            continue
        child_requirements = []
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            child_name = canonicalize_name(requirement.name)
            child_requirements.append(str(requirement))
            pending.append((child_name, str(requirement)))
        license_value = dist.metadata.get("License-Expression") or dist.metadata.get("License") or "unknown"
        if not str(license_value).strip() or str(license_value).strip().upper() == "UNKNOWN":
            license_value = "unknown"
        resolved[name] = {
            "name": name,
            "version": dist.version,
            "scope": "direct" if name in direct_names else "transitive",
            "declared_requirement": declaration if name in direct_names else None,
            "dependencies": sorted(set(child_requirements)),
            "ecosystem": "pypi",
            "license": str(license_value).strip(),
            "artifact_sha256": "unknown",
        }
    return {
        "schema_version": "1",
        "kind": "python_installed_metadata_lock_v1",
        "direct_requirements": direct_lines,
        "packages": [resolved[name] for name in sorted(resolved)],
        "missing_packages": sorted(missing),
        "unpinned_direct_requirements": sorted(unpinned_direct_requirements),
        "complete": not missing,
    }


def build_node_sbom(package_lock_path: Path) -> dict[str, Any]:
    value = json.loads(package_lock_path.read_text(encoding="utf-8"))
    if value.get("lockfileVersion") != 3 or not isinstance(value.get("packages"), dict):
        raise ReleaseCandidateNotReady("node_lock_format_unsupported")
    root = value["packages"].get("") or {}
    direct_runtime = set((root.get("dependencies") or {}).keys())
    direct_development = set((root.get("devDependencies") or {}).keys())
    packages = []
    for path, item in sorted(value["packages"].items()):
        if not path:
            continue
        name = path.rsplit("node_modules/", 1)[-1]
        direct = name in direct_runtime or name in direct_development
        scope = "runtime" if not item.get("dev") else "development"
        packages.append({
            "name": name,
            "version": item.get("version") or "unknown",
            "scope": scope,
            "direct": direct,
            "ecosystem": "npm",
            "license": item.get("license") or "unknown",
            "resolved": item.get("resolved") or "unknown",
            "integrity": item.get("integrity") or "unknown",
            "dependencies": sorted((item.get("dependencies") or {}).keys()),
        })
    return {
        "schema_version": "1",
        "kind": "npm_package_lock_v3",
        "lockfile_version": 3,
        "packages": packages,
        "package_count": len(packages),
        "unknown_license_count": sum(item["license"] == "unknown" for item in packages),
    }


def freeze_contract(root: Path, spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = source_manifest_from_git(root, spec)
    python_lock = build_python_lock(root / "requirements.txt")
    node_sbom = build_node_sbom(root / "frontend/package-lock.json")
    toolchain = _toolchain_summary()
    dependency_files = {}
    for relative in spec["dependency_inputs"]:
        path = root / relative
        if not path.is_file():
            raise ReleaseCandidateNotReady(f"dependency_input_missing:{relative}")
        dependency_files[relative] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    overlay_template = "\n".join(spec["frontend_build_overlay"]["template_lines"]) + "\n"
    source_commit = str(spec["source_commit"])
    next_build_seeds = {
        "preview_mode_id": hashlib.sha256(f"preview-id:{source_commit}".encode()).hexdigest()[:32],
        "preview_signing_key": hashlib.sha256(f"preview-sign:{source_commit}".encode()).hexdigest(),
        "preview_encryption_key": hashlib.sha256(f"preview-encrypt:{source_commit}".encode()).hexdigest(),
        "server_actions_encryption_key": base64.b64encode(
            hashlib.sha256(f"server-actions:{source_commit}".encode()).digest()
        ).decode(),
        "expire_at": 4102444800000,
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": spec["source_commit"],
        "source_date_epoch": spec["source_date_epoch"],
        "source_manifest": source_manifest,
        "source_manifest_sha256": stable_digest(source_manifest),
        "toolchain": toolchain,
        "toolchain_requirements": spec["toolchain_requirements"],
        "dependency_inputs": dependency_files,
        "python_lock": {
            "path": "benchmark/release_candidate_reproducibility_v1_python_lock.json",
            "sha256": stable_digest(python_lock),
        },
        "node_sbom_sha256": stable_digest(node_sbom),
        "allowed_environment": spec["allowed_environment"],
        "build_commands": spec["build_commands"],
        "frontend_build_overlay": {
            **spec["frontend_build_overlay"],
            "content": overlay_template.replace(
                "{commit}", str(spec["source_commit"])[:20]
            ),
        },
        "next_build_seeds": next_build_seeds,
        "archive": spec["archive"],
        "output_names": spec["output_names"],
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }
    contract["frontend_build_overlay"]["sha256"] = sha256_bytes(contract["frontend_build_overlay"]["content"].encode())
    return contract, {"python": python_lock, "node": node_sbom}


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseCandidateNotReady("contract_unavailable") from exc
    if value.get("schema_version") != SCHEMA_VERSION or value.get("protocol") != PROTOCOL:
        raise ReleaseCandidateError("contract_version_mismatch")
    if value.get("execution") != EXECUTION or value.get("formal_validation_complete") is not False:
        raise ReleaseCandidateError("contract_boundary_drift")
    rows = value.get("source_manifest")
    if not isinstance(rows, list) or not rows:
        raise ReleaseCandidateError("source_manifest_invalid")
    paths = [str(item.get("path")) for item in rows]
    if len(paths) != len(set(paths)):
        raise ReleaseCandidateError("source_manifest_duplicate")
    if rows != sorted(rows, key=lambda item: item["path"]):
        raise ReleaseCandidateError("source_manifest_invalid")
    for path in paths:
        _safe_relative(path)
    if stable_digest(rows) != value.get("source_manifest_sha256"):
        raise ReleaseCandidateError("source_manifest_digest_mismatch")
    if str(_git(root, "rev-parse", "HEAD")).strip() != value["source_commit"]:
        raise ReleaseCandidateNotReady("source_head_mismatch")
    return value


def _toolchain_summary() -> dict[str, str]:
    import setuptools
    import wheel

    node = subprocess.run(["node", "--version"], check=False, capture_output=True, text=True, timeout=10)
    npm = subprocess.run(["npm", "--version"], check=False, capture_output=True, text=True, timeout=10)
    if node.returncode or npm.returncode:
        raise ReleaseCandidateNotReady("node_toolchain_missing")
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "python_implementation": sys.implementation.name,
        "setuptools": setuptools.__version__,
        "wheel": wheel.__version__,
        "node": node.stdout.strip().removeprefix("v"),
        "npm": npm.stdout.strip(),
    }


def _verify_toolchain(contract: Mapping[str, Any]) -> None:
    actual = _toolchain_summary()
    expected = contract["toolchain"]
    if actual != expected:
        raise ReleaseCandidateNotReady("toolchain_drift")
    requirements = contract["toolchain_requirements"]
    if not actual["python"].startswith(str(requirements["python_major_minor"]) + "."):
        raise ReleaseCandidateNotReady("python_version_unsupported")
    if int(actual["node"].split(".")[0]) != requirements["node_major"]:
        raise ReleaseCandidateNotReady("node_version_unsupported")
    if int(actual["npm"].split(".")[0]) != requirements["npm_major"]:
        raise ReleaseCandidateNotReady("npm_version_unsupported")


def materialize_source(root: Path, contract: Mapping[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    commit = str(contract["source_commit"])
    for item in contract["source_manifest"]:
        relative = _safe_relative(str(item["path"]))
        content = _git(root, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(content, bytes)
        if len(content) != item["size"] or sha256_bytes(content) != item["sha256"]:
            raise ReleaseCandidateError("source_blob_mismatch")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        output.chmod(0o644)


def _zip_info(name: str, epoch: int) -> zipfile.ZipInfo:
    import datetime

    stamp = datetime.datetime.fromtimestamp(max(epoch, 315532800), datetime.UTC)
    info = zipfile.ZipInfo(name, (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.create_system = 3
    return info


def build_wheel(source_root: Path, output: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    epoch = int(contract["source_date_epoch"])
    members: dict[str, bytes] = {}
    for item in contract["source_manifest"]:
        relative = str(item["path"])
        if relative.startswith("src/scholar_agent/") and not relative.endswith((".pyc", ".DS_Store")):
            members[relative.removeprefix("src/")] = (source_root / relative).read_bytes()
    dependency_contract = contract.get("python_dependency_lock")
    requirements = (
        list(dependency_contract["runtime_requires_dist"])
        if dependency_contract is not None
        else [
            line.strip()
            for line in (source_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    )
    metadata_text = "\n".join([
        "Metadata-Version: 2.3",
        "Name: spar-scholar-agent",
        "Version: 0.1.0",
        "Summary: SPAR scholarly retrieval agent",
        "Requires-Python: >=3.11",
        *[f"Requires-Dist: {item}" for item in requirements],
        "",
    ]).encode()
    members[f"{DIST_INFO}/METADATA"] = metadata_text
    members[f"{DIST_INFO}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: release_candidate_reproducibility_v1\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    members[f"{DIST_INFO}/top_level.txt"] = b"scholar_agent\n"
    record_rows = []
    for name, content in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        record_rows.append((name, f"sha256={digest}", str(len(content))))
    record_rows.append((f"{DIST_INFO}/RECORD", "", ""))
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(record_rows)
    members[f"{DIST_INFO}/RECORD"] = record_buffer.getvalue().encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(_zip_info(name, epoch), content)
    return {"path": output.name, "size": output.stat().st_size, "sha256": sha256_file(output), "member_count": len(members)}


def _tar_bytes(files: Mapping[str, bytes], contract: Mapping[str, Any]) -> bytes:
    raw = io.BytesIO()
    epoch = int(contract["source_date_epoch"])
    archive_config = contract["archive"]
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in sorted(files.items()):
            relative = _safe_relative(name)
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mtime = epoch
            info.mode = int(archive_config["file_mode"])
            info.uid = int(archive_config["uid"])
            info.gid = int(archive_config["gid"])
            info.uname = str(archive_config["uname"])
            info.gname = str(archive_config["gname"])
            archive.addfile(info, io.BytesIO(content))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=epoch, compresslevel=9) as handle:
        handle.write(raw.getvalue())
    return compressed.getvalue()


def build_source_archive(source_root: Path, output: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    files = {str(item["path"]): (source_root / str(item["path"])).read_bytes() for item in contract["source_manifest"]}
    output.write_bytes(_tar_bytes(files, contract))
    return {"path": output.name, "size": output.stat().st_size, "sha256": sha256_file(output), "member_count": len(files)}


def _minimal_build_env(home: Path, temp: Path, contract: Mapping[str, Any]) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "NEXT_TELEMETRY_DISABLED": "1",
        "NEXT_SERVER_ACTIONS_ENCRYPTION_KEY": str(
            contract["next_build_seeds"]["server_actions_encryption_key"]
        ),
        "PATH": os.environ.get("PATH", ""),
        "SOURCE_DATE_EPOCH": str(contract["source_date_epoch"]),
        "TMPDIR": str(temp),
        "TZ": "UTC",
    }


def build_frontend(source_root: Path, output: Path, contract: Mapping[str, Any], repository_root: Path, profile_root: Path) -> dict[str, Any]:
    staging = contract.get("frontend_canonical_staging")
    canonical_parent: Path | None = None
    lock_path: Path | None = None
    effective_source = source_root
    if staging:
        if staging.get("strategy") != "posix_tmp_fixed_source_digest_v1":
            raise ReleaseCandidateError("frontend_staging_strategy_unsupported")
        digest = str(contract["source_manifest_sha256"])
        canonical_parent = Path("/tmp") / str(staging["namespace"]) / digest[:24]
        lock_path = canonical_parent.with_name(canonical_parent.name + ".lock")
        canonical_parent.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock_path.mkdir()
        except FileExistsError as exc:
            raise ReleaseCandidateNotReady("frontend_canonical_staging_busy") from exc
        shutil.rmtree(canonical_parent, ignore_errors=True)
        effective_source = canonical_parent / "source"
        shutil.copytree(source_root, effective_source, copy_function=shutil.copyfile)
    frontend = effective_source / "frontend"
    overlay = contract["frontend_build_overlay"]
    if sha256_bytes(str(overlay["content"]).encode()) != overlay["sha256"]:
        raise ReleaseCandidateError("frontend_overlay_hash_mismatch")
    (frontend / "next.config.ts").write_text(str(overlay["content"]), encoding="utf-8")
    modules = repository_root / "frontend/node_modules"
    if not modules.is_dir():
        raise ReleaseCandidateNotReady("offline_node_modules_missing")
    (frontend / "node_modules").symlink_to(modules, target_is_directory=True)
    preview = contract["next_build_seeds"]
    preview_cache = frontend / ".next/cache"
    preview_cache.mkdir(parents=True)
    (preview_cache / ".previewinfo").write_text(
        json.dumps(
            {
                "previewModeId": preview["preview_mode_id"],
                "previewModeSigningKey": preview["preview_signing_key"],
                "previewModeEncryptionKey": preview["preview_encryption_key"],
                "expireAt": preview["expire_at"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    home = profile_root / "home"
    temp = profile_root / "tmp"
    home.mkdir(parents=True)
    temp.mkdir(parents=True)
    next_cli = modules / ".bin/next"
    try:
        completed = subprocess.run(
            [str(next_cli), "build", "--webpack"],
            cwd=frontend,
            env=_minimal_build_env(home, temp, contract),
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            raise ReleaseCandidateNotReady("offline_frontend_build_failed")
        build = frontend / ".next"
        required = [build / "BUILD_ID", build / "static", build / "server/app"]
        if any(not item.exists() for item in required):
            raise ReleaseCandidateNotReady("frontend_static_output_missing")
        files: dict[str, bytes] = {}
        allowed_names = {
            "BUILD_ID", "app-build-manifest.json", "build-manifest.json", "prerender-manifest.json", "routes-manifest.json"
        }
        for path in build.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(build).as_posix()
            if relative in allowed_names or relative.startswith("static/") or (
                relative.startswith("server/app/") and path.suffix in {".html", ".rsc", ".txt", ".json"}
            ):
                files[f"frontend/{relative}"] = path.read_bytes()
        for path in (frontend / "public").rglob("*") if (frontend / "public").is_dir() else []:
            if path.is_file():
                files[f"frontend/public/{path.relative_to(frontend / 'public').as_posix()}"] = path.read_bytes()
        if not files:
            raise ReleaseCandidateNotReady("frontend_static_member_set_empty")
        forbidden = [
            str(source_root).encode(), str(effective_source).encode(), str(repository_root).encode(),
            str(home).encode(), str(temp).encode(),
        ]
        if any(token and token in content for content in files.values() for token in forbidden):
            raise ReleaseCandidateError("absolute_path_embedded_in_frontend")
        output.write_bytes(_tar_bytes(files, contract))
        return {"path": output.name, "size": output.stat().st_size, "sha256": sha256_file(output), "member_count": len(files)}
    finally:
        if canonical_parent is not None:
            shutil.rmtree(canonical_parent, ignore_errors=True)
        if lock_path is not None:
            shutil.rmtree(lock_path, ignore_errors=True)


def _dependency_report(root: Path, contract: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    python_lock = build_python_lock(root / "requirements.txt")
    node = build_node_sbom(root / "frontend/package-lock.json")
    violations = []
    if stable_digest(python_lock) != contract["python_lock"]["sha256"]:
        violations.append("python_dependency_lock_drift")
    tracked_python_lock = root / str(contract["python_lock"]["path"])
    if not tracked_python_lock.is_file():
        violations.append("python_dependency_lock_missing")
    else:
        try:
            tracked_value = json.loads(tracked_python_lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            violations.append("python_dependency_lock_invalid")
        else:
            if stable_digest(tracked_value) != contract["python_lock"]["sha256"]:
                violations.append("python_dependency_lock_artifact_drift")
    if stable_digest(node) != contract["node_sbom_sha256"]:
        violations.append("node_dependency_lock_drift")
    if not python_lock["complete"]:
        violations.append("python_offline_dependency_missing")
    if python_lock["unpinned_direct_requirements"]:
        violations.append("python_direct_requirements_unpinned")
    dependency_contract = contract.get("python_dependency_lock")
    if dependency_contract is not None:
        if dependency_contract.get("protocol") != "python_dependency_lock_v1":
            violations.append("python_dependency_contract_version_mismatch")
        if dependency_contract.get("lock_qualified") is not True:
            violations.append("python_dependency_lock_not_qualified")
        if dependency_contract.get("offline_install_qualified") is not True:
            violations.append("python_offline_install_not_qualified")
    wheelhouse_contract = contract.get("offline_wheelhouse_intake")
    if wheelhouse_contract is not None:
        if wheelhouse_contract.get("protocol") != "offline_wheelhouse_intake_v1":
            violations.append("python_wheelhouse_contract_version_mismatch")
        manifest_path = root / str(wheelhouse_contract.get("manifest_path") or "")
        if not manifest_path.is_file():
            violations.append("python_wheelhouse_manifest_missing")
        elif sha256_file(manifest_path) != wheelhouse_contract.get("manifest_sha256"):
            violations.append("python_wheelhouse_manifest_drift")
        if wheelhouse_contract.get("real_wheelhouse_qualified") is not True:
            violations.append("python_offline_wheelhouse_not_qualified")
    for relative, expected in contract["dependency_inputs"].items():
        path = root / relative
        if not path.is_file():
            violations.append(f"dependency_input_missing:{relative}")
        elif sha256_file(path) != expected["sha256"]:
            violations.append(f"dependency_input_drift:{relative}")
    sbom = {
        "schema_version": "1",
        "protocol": PROTOCOL,
        "python": python_lock,
        "node": node,
        "summary": {
            "python_package_count": len(python_lock["packages"]),
            "python_unknown_license_count": sum(item["license"] == "unknown" for item in python_lock["packages"]),
            "python_unpinned_declaration_count": len(python_lock["unpinned_direct_requirements"]),
            "node_package_count": node["package_count"],
            "node_unknown_license_count": node["unknown_license_count"],
            "development_dependency_files_in_runtime_package": 0,
        },
    }
    return sbom, sorted(violations)


def build_once(repository_root: Path, contract: Mapping[str, Any], output_dir: Path, profile_root: Path) -> dict[str, Any]:
    _verify_toolchain(contract)
    source = profile_root / "source"
    materialize_source(repository_root, contract, source)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = []
    artifacts.append(build_wheel(source, output_dir / WHEEL_NAME, contract))
    artifacts.append(build_frontend(source, output_dir / "frontend-static.tar.gz", contract, repository_root, profile_root))
    artifacts.append(build_source_archive(source, output_dir / "source.tar.gz", contract))
    sbom, dependency_violations = _dependency_report(repository_root, contract)
    sbom_bytes = canonical_json(sbom)
    (output_dir / "sbom.json").write_bytes(sbom_bytes)
    artifacts.append({"path": "sbom.json", "size": len(sbom_bytes), "sha256": sha256_bytes(sbom_bytes)})
    manifest = {
        "schema_version": "1",
        "protocol": PROTOCOL,
        "source_commit": contract["source_commit"],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "toolchain": contract["toolchain"],
        "build_commands": contract["build_commands"],
        "allowed_environment": contract["allowed_environment"],
        "frontend_build_overlay_sha256": contract["frontend_build_overlay"]["sha256"],
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "dependency_violations": dependency_violations,
        "execution": EXECUTION,
        "quality_metrics_included": False,
        "official_submission_included": False,
    }
    manifest_bytes = canonical_json(manifest)
    (output_dir / "release-manifest.json").write_bytes(manifest_bytes)
    inner = {
        item.name: item.read_bytes()
        for item in sorted(output_dir.iterdir())
        if item.is_file() and item.name != "spar-release-candidate.tar.gz"
    }
    (output_dir / "spar-release-candidate.tar.gz").write_bytes(_tar_bytes(inner, contract))
    manifest["release_archive"] = {
        "path": "spar-release-candidate.tar.gz",
        "size": (output_dir / "spar-release-candidate.tar.gz").stat().st_size,
        "sha256": sha256_file(output_dir / "spar-release-candidate.tar.gz"),
    }
    return manifest


def verify_output(output_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = set(contract["output_names"])
    actual = {item.name for item in output_dir.iterdir() if item.is_file()}
    violations = []
    if actual != expected:
        violations.append({"invariant": "output_member_set", "expected": sorted(expected), "actual": sorted(actual)})
    manifest_path = output_dir / "release-manifest.json"
    if not manifest_path.is_file():
        raise ReleaseCandidateNotReady("release_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("artifacts") or []:
        path = output_dir / _safe_relative(str(item["path"]))
        if not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            violations.append({"invariant": "artifact_integrity", "path": item["path"]})
    for name in (WHEEL_NAME, "frontend-static.tar.gz", "source.tar.gz", "spar-release-candidate.tar.gz"):
        path = output_dir / name
        if path.is_file():
            try:
                if name.endswith(".whl"):
                    with zipfile.ZipFile(path) as archive:
                        members = archive.namelist()
                    if "scholar_agent/__init__.py" not in members or f"{DIST_INFO}/METADATA" not in members:
                        violations.append({"invariant": "wheel_entrypoint_members", "path": name})
                else:
                    with tarfile.open(path, "r:gz") as archive:
                        members = archive.getnames()
                if len(members) != len(set(members)) or any(_safe_relative(member) != member for member in members):
                    violations.append({"invariant": "archive_member_safety", "path": name})
            except (tarfile.TarError, zipfile.BadZipFile, OSError):
                violations.append({"invariant": "archive_readable", "path": name})
    return _report("reproducible_release_ready" if not violations else "build_or_supply_chain_violation", EXIT_READY if not violations else EXIT_VIOLATION, violations=violations, manifest=manifest)


def compare_outputs(first: Path, second: Path) -> dict[str, Any]:
    first_files = {item.name: sha256_file(item) for item in first.iterdir() if item.is_file()}
    second_files = {item.name: sha256_file(item) for item in second.iterdir() if item.is_file()}
    differences = [
        {"path": name, "first_sha256": first_files.get(name), "second_sha256": second_files.get(name)}
        for name in sorted(set(first_files) | set(second_files))
        if first_files.get(name) != second_files.get(name)
    ]
    return _report(
        "reproducible_release_ready" if not differences else "build_or_supply_chain_violation",
        EXIT_READY if not differences else EXIT_VIOLATION,
        differences=differences,
        artifact_count=len(first_files) if not differences else None,
    )


def double_build(repository_root: Path, contract: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    profiles = []
    for name in ("profile-a", "different-parent/profile-b"):
        profile = output_root / name
        profile.mkdir(parents=True)
        outputs = profile / "outputs"
        manifest = build_once(repository_root, contract, outputs, profile)
        profiles.append({"profile": name, "outputs": outputs, "manifest": manifest})
    comparison = compare_outputs(profiles[0]["outputs"], profiles[1]["outputs"])
    verification = [verify_output(item["outputs"], contract) for item in profiles]
    dependency_violations = sorted(set(profiles[0]["manifest"]["dependency_violations"] + profiles[1]["manifest"]["dependency_violations"]))
    ready = comparison["exit_code"] == 0 and all(item["exit_code"] == 0 for item in verification) and not dependency_violations
    missing_dependency = any(value.endswith("_missing") for value in dependency_violations)
    status = "reproducible_release_ready" if ready else (
        "not_ready_missing_offline_dependency_or_input"
        if missing_dependency and comparison["exit_code"] == 0
        else "build_or_supply_chain_violation"
    )
    code = EXIT_READY if ready else (
        EXIT_NOT_READY if status == "not_ready_missing_offline_dependency_or_input" else EXIT_VIOLATION
    )
    return _report(
        status,
        code,
        source_commit=contract["source_commit"],
        comparison=comparison,
        verification=verification,
        dependency_violations=dependency_violations,
        artifacts=profiles[0]["manifest"]["artifacts"],
        release_archive=profiles[0]["manifest"]["release_archive"],
        sbom_summary=json.loads((profiles[0]["outputs"] / "sbom.json").read_text())["summary"],
    )


def audit_readiness(repository_root: Path, contract: Mapping[str, Any], evidence_path: Path | None = None) -> dict[str, Any]:
    violations = []
    try:
        _verify_toolchain(contract)
        sbom, dependency_violations = _dependency_report(repository_root, contract)
    except ReleaseCandidateNotReady as exc:
        return _report("not_ready_missing_offline_dependency_or_input", EXIT_NOT_READY, reason=str(exc))
    if evidence_path:
        if not evidence_path.is_file():
            return _report("not_ready_missing_offline_dependency_or_input", EXIT_NOT_READY, reason="double_build_evidence_missing")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("protocol") != PROTOCOL:
            violations.append("evidence_protocol_mismatch")
        if evidence.get("source_commit") != contract["source_commit"]:
            violations.append("evidence_source_commit_mismatch")
        if evidence.get("status") != "reproducible_release_ready":
            violations.append("release_candidate_not_reproducible")
    if violations:
        return _report("build_or_supply_chain_violation", EXIT_VIOLATION, violations=violations)
    if dependency_violations:
        missing = any(value.endswith("_missing") for value in dependency_violations)
        return _report(
            "not_ready_missing_offline_dependency_or_input" if missing else "build_or_supply_chain_violation",
            EXIT_NOT_READY if missing else EXIT_VIOLATION,
            dependency_violations=dependency_violations,
            sbom_summary=sbom["summary"],
        )
    return _report("reproducible_release_ready", EXIT_READY, sbom_summary=sbom["summary"])


def summarize_double_build_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Create stable tracked evidence without preserving non-deterministic hashes."""

    differences = sorted(
        str(item["path"])
        for item in (report.get("comparison") or {}).get("differences") or []
    )
    stable_artifacts = [
        {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
        for item in report.get("artifacts") or []
        if item["path"] not in set(differences)
    ]
    qualified = report.get("status") == "reproducible_release_ready" and not differences
    return _report(
        str(report.get("status")),
        int(report.get("exit_code", EXIT_VIOLATION)),
        source_commit=report.get("source_commit"),
        build_profile_count=2,
        differing_artifact_paths=differences,
        reproducible_artifacts=sorted(stable_artifacts, key=lambda item: item["path"]),
        reproducible_artifact_count=len(stable_artifacts),
        sbom_summary=report.get("sbom_summary"),
        dependency_violations=sorted(report.get("dependency_violations") or []),
        qualification="qualified" if qualified else "not_qualified",
        limitation=(
            "frontend_webpack_output_not_byte_reproducible_across_isolated_source_paths"
            if "frontend-static.tar.gz" in differences
            else None
        ),
    )


def _report(status: str, exit_code: int, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "execution": EXECUTION,
        "formal_validation_complete": False,
        **extra,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
