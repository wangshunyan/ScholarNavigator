"""Strict, deterministic intake for an offline Python wheelhouse."""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from collections import defaultdict
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from scholar_agent.evaluation.release_candidate_reproducibility import (
    EXECUTION,
    canonical_json,
    freeze_contract,
    sha256_bytes,
    sha256_file,
    stable_digest,
)


PROTOCOL = "offline_wheelhouse_intake_v1"
SCHEMA_VERSION = "1"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
_ENTRY_POINT = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*(?::[A-Za-z_][A-Za-z0-9_.]*)?(?:\[[A-Za-z0-9_, -]+\])?$"
)


class WheelhouseError(RuntimeError):
    """The artifact set or its declared contract is invalid."""


class WheelhouseNotReady(WheelhouseError):
    """Required offline artifacts have not been supplied."""


def _safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WheelhouseError("unsafe_wheel_member_path")
    return path.as_posix()


def _record_hash(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def _metadata_license(message: Mapping[str, Any]) -> dict[str, str]:
    value = str(
        message.get("License-Expression") or message.get("License") or ""
    ).strip()
    if not value or value.upper() == "UNKNOWN":
        return {"status": "unknown", "value": "unknown"}
    return {"status": "declared_in_wheel_metadata", "value": value}


def _validate_entry_points(archive: zipfile.ZipFile, members: set[str]) -> list[str]:
    paths = sorted(name for name in members if name.endswith(".dist-info/entry_points.txt"))
    if not paths:
        return []
    if len(paths) != 1:
        raise WheelhouseError("entry_point_metadata_ambiguous")
    try:
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(archive.read(paths[0]).decode("utf-8"))
    except (UnicodeError, configparser.Error) as exc:
        raise WheelhouseError("entry_point_metadata_invalid") from exc
    values: list[str] = []
    for section in parser.sections():
        if not section or any(ord(char) < 32 for char in section):
            raise WheelhouseError("entry_point_group_invalid")
        for name, value in parser.items(section):
            normalized = value.strip()
            if (
                not name.strip()
                or any(ord(char) < 32 for char in name)
                or not _ENTRY_POINT.fullmatch(normalized)
            ):
                raise WheelhouseError("entry_point_invalid")
            values.append(f"{section}:{name.strip()}={normalized}")
    return sorted(values)


def inspect_wheel(path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one wheel without trusting its filename or extracting it."""

    if path.suffix != ".whl":
        raise WheelhouseError("sdist_or_non_wheel_rejected")
    try:
        parsed_name, parsed_version, _build, filename_tags = parse_wheel_filename(path.name)
    except ValueError as exc:
        raise WheelhouseError("wheel_filename_invalid") from exc
    if not set(filename_tags).intersection(sys_tags()):
        raise WheelhouseError("wheel_tag_incompatible")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise WheelhouseError("wheel_zip_invalid") from exc
    with archive:
        infos = archive.infolist()
        names = [_safe_member(info.filename.rstrip("/")) for info in infos]
        if len(names) != len(set(names)):
            raise WheelhouseError("wheel_duplicate_member")
        if len(infos) > int(policy["max_file_count"]):
            raise WheelhouseError("wheel_file_count_limit")
        total = 0
        members = set(names)
        for info, name in zip(infos, names, strict=True):
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise WheelhouseError("wheel_link_member_rejected")
            if info.is_dir():
                continue
            total += info.file_size
            if info.file_size > int(policy["max_member_uncompressed_bytes"]):
                raise WheelhouseError("wheel_member_size_limit")
            if total > int(policy["max_total_uncompressed_bytes"]):
                raise WheelhouseError("wheel_total_size_limit")
            if info.file_size and info.compress_size == 0:
                raise WheelhouseError("wheel_compression_ratio_limit")
            if info.compress_size and info.file_size / info.compress_size > float(
                policy["max_compression_ratio"]
            ):
                raise WheelhouseError("wheel_compression_ratio_limit")
            _safe_member(name)

        metadata_paths = sorted(
            name for name in members if name.endswith(".dist-info/METADATA")
        )
        wheel_paths = sorted(name for name in members if name.endswith(".dist-info/WHEEL"))
        record_paths = sorted(name for name in members if name.endswith(".dist-info/RECORD"))
        if len(metadata_paths) != 1 or len(wheel_paths) != 1 or len(record_paths) != 1:
            raise WheelhouseError("wheel_dist_info_incomplete_or_ambiguous")
        roots = {
            value.rsplit("/", 1)[0]
            for value in (metadata_paths[0], wheel_paths[0], record_paths[0])
        }
        if len(roots) != 1:
            raise WheelhouseError("wheel_dist_info_mixed")
        try:
            metadata_bytes = archive.read(metadata_paths[0])
            message = Parser().parsestr(metadata_bytes.decode("utf-8"))
        except (KeyError, UnicodeError) as exc:
            raise WheelhouseError("wheel_metadata_invalid") from exc
        metadata_name = canonicalize_name(str(message.get("Name") or ""))
        metadata_version = str(message.get("Version") or "")
        if metadata_name != canonicalize_name(str(parsed_name)):
            raise WheelhouseError("wheel_name_metadata_mismatch")
        try:
            version_matches = Version(metadata_version) == parsed_version
        except ValueError as exc:
            raise WheelhouseError("wheel_version_metadata_invalid") from exc
        if not version_matches:
            raise WheelhouseError("wheel_version_metadata_mismatch")

        wheel_message = Parser().parsestr(archive.read(wheel_paths[0]).decode("utf-8"))
        metadata_tags = set(wheel_message.get_all("Tag") or [])
        if metadata_tags != {str(tag) for tag in filename_tags}:
            raise WheelhouseError("wheel_tag_metadata_mismatch")

        try:
            record_rows = list(
                csv.reader(io.StringIO(archive.read(record_paths[0]).decode("utf-8")))
            )
        except (UnicodeError, csv.Error) as exc:
            raise WheelhouseError("wheel_record_invalid") from exc
        record: dict[str, tuple[str, str]] = {}
        for row in record_rows:
            if len(row) != 3:
                raise WheelhouseError("wheel_record_invalid")
            name = _safe_member(row[0])
            if name in record:
                raise WheelhouseError("wheel_record_duplicate")
            record[name] = (row[1], row[2])
        file_members = {name for info, name in zip(infos, names, strict=True) if not info.is_dir()}
        if set(record) != file_members:
            raise WheelhouseError("wheel_record_member_set_mismatch")
        for name in sorted(file_members):
            digest, size = record[name]
            if name == record_paths[0]:
                if digest or size:
                    raise WheelhouseError("wheel_record_self_entry_invalid")
                continue
            content = archive.read(name)
            if not digest or digest != _record_hash(content):
                raise WheelhouseError("wheel_record_hash_invalid")
            if size != str(len(content)):
                raise WheelhouseError("wheel_record_size_invalid")

        requires_dist = sorted(message.get_all("Requires-Dist") or [])
        for raw in requires_dist:
            try:
                Requirement(raw)
            except ValueError as exc:
                raise WheelhouseError("wheel_requires_dist_invalid") from exc
        entry_points = _validate_entry_points(archive, members)
    return {
        "filename": path.name,
        "name": metadata_name,
        "version": metadata_version,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "python_abi_platform_tags": sorted(str(tag) for tag in filename_tags),
        "requires_dist": requires_dist,
        "license_evidence": _metadata_license(message),
        "source_type": "unknown",
        "entry_points": entry_points,
        "member_count": len(infos),
        "uncompressed_size": total,
        "metadata_sha256": sha256_bytes(metadata_bytes),
    }


def _active_dependencies(rows: Sequence[str]) -> list[Requirement]:
    environment = default_environment()
    active = []
    for raw in rows:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        active.append(requirement)
    return active


def _validate_closure(
    artifacts: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[str]]:
    locked = {item["name"]: item for item in lock["packages"]}
    graph: dict[str, set[str]] = defaultdict(set)
    violations: list[str] = []
    for artifact in artifacts:
        parent = str(artifact["name"])
        for requirement in _active_dependencies(artifact["requires_dist"]):
            child = canonicalize_name(requirement.name)
            expected = locked.get(child)
            if expected is None:
                violations.append(f"dependency_not_locked:{parent}:{child}")
                continue
            if requirement.specifier and not requirement.specifier.contains(
                expected["version"], prereleases=True
            ):
                violations.append(f"dependency_version_conflict:{parent}:{child}")
                continue
            graph[parent].add(child)
    for item in lock["packages"]:
        name = str(item["name"])
        if sorted(graph.get(name, set())) != sorted(item["dependencies"]):
            violations.append(f"dependency_closure_drift:{name}")
    edges = [
        {"from": parent, "to": child}
        for parent in sorted(graph)
        for child in sorted(graph[parent])
    ]
    return edges, sorted(set(violations))


def build_manifest(
    wheelhouse: Path,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if protocol.get("protocol") != PROTOCOL or protocol.get("schema_version") != SCHEMA_VERSION:
        raise WheelhouseError("protocol_version_mismatch")
    lock_sha256 = sha256_bytes(canonical_json(lock))
    if lock_sha256 != protocol["dependency_lock"]["sha256"]:
        raise WheelhouseError("dependency_lock_hash_mismatch")
    expected = {(item["name"], item["version"]): item for item in lock["packages"]}
    paths = sorted(wheelhouse.iterdir(), key=lambda path: path.name) if wheelhouse.is_dir() else []
    if any(path.is_dir() or path.suffix != ".whl" for path in paths):
        raise WheelhouseError("sdist_extra_or_directory_rejected")
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        artifact = inspect_wheel(path, protocol["intake"])
        key = (artifact["name"], artifact["version"])
        if key not in expected:
            raise WheelhouseError("extra_or_version_conflicting_wheel")
        if key in seen:
            raise WheelhouseError("duplicate_locked_wheel")
        seen.add(key)
        artifact["groups"] = sorted(expected[key]["groups"])
        artifacts.append(artifact)
    missing = [
        {"name": name, "version": version, "groups": sorted(item["groups"])}
        for (name, version), item in sorted(expected.items())
        if (name, version) not in seen
    ]
    edges, closure_violations = _validate_closure(artifacts, lock) if not missing else ([], [])
    violations = closure_violations
    if violations:
        status, exit_code = "artifact_or_supply_chain_violation", EXIT_VIOLATION
    elif missing:
        status, exit_code = "not_ready_missing_required_wheels", EXIT_NOT_READY
    else:
        status, exit_code = "wheelhouse_qualified", EXIT_QUALIFIED
    index = [
        {
            key: artifact[key]
            for key in (
                "filename",
                "groups",
                "license_evidence",
                "metadata_sha256",
                "name",
                "python_abi_platform_tags",
                "requires_dist",
                "sha256",
                "size",
                "source_type",
                "version",
            )
        }
        for artifact in sorted(artifacts, key=lambda item: (item["name"], item["filename"]))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "source_commit": protocol["source_commit"],
        "dependency_lock": {
            "protocol": lock["protocol"],
            "sha256": lock_sha256,
            "package_count": lock["package_count"],
        },
        "status": status,
        "exit_code": exit_code,
        "expected_wheel_count": len(expected),
        "accepted_wheel_count": len(artifacts),
        "missing_wheels": missing,
        "violations": violations,
        "dependency_edges": edges,
        "wheelhouse_index": index,
        "install_plan": install_plan(index) if not missing and not violations else [],
        "real_wheelhouse_qualified": exit_code == EXIT_QUALIFIED,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def install_plan(index: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}"
        for item in sorted(index, key=lambda item: item["name"])
    ]


def verify_manifest(
    wheelhouse: Path,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    tracked: Mapping[str, Any],
) -> dict[str, Any]:
    actual = build_manifest(wheelhouse, lock, protocol)
    violations = list(actual["violations"])
    if actual != tracked:
        violations.append("wheelhouse_manifest_drift")
    if violations:
        status, exit_code = "artifact_or_supply_chain_violation", EXIT_VIOLATION
    else:
        status, exit_code = actual["status"], actual["exit_code"]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "accepted_wheel_count": actual["accepted_wheel_count"],
        "expected_wheel_count": actual["expected_wheel_count"],
        "missing_wheels": actual["missing_wheels"],
        "violations": sorted(set(violations)),
        "manifest_sha256": stable_digest(actual),
        "real_wheelhouse_qualified": actual["real_wheelhouse_qualified"] and not violations,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def _run_install_profiles(
    wheelhouse: Path,
    plan: Sequence[str],
    package_names: Sequence[str],
    *,
    smoke_code: str,
    cli: Sequence[str],
    residue_modules: Sequence[str],
    source_root: Path | None = None,
) -> list[dict[str, Any]]:
    results = []
    with tempfile.TemporaryDirectory(prefix="wheelhouse-install-") as temporary:
        root = Path(temporary)
        plan_path = root / "requirements.lock"
        plan_path.write_text("\n".join(plan) + "\n", encoding="utf-8")
        for index in range(2):
            profile = root / f"profile-{index}"
            home, tmp, environment = profile / "home", profile / "tmp", profile / "venv"
            home.mkdir(parents=True)
            tmp.mkdir()
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / "bin/python"
            env = {
                "HOME": str(home),
                "TMPDIR": str(tmp),
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": str(environment / "bin"),
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
            if source_root is not None:
                env["PYTHONPATH"] = str(source_root / "src")
            install = subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--require-hashes",
                    "--find-links",
                    str(wheelhouse),
                    "-r",
                    str(plan_path),
                ],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            smoke = subprocess.run(
                [str(python), "-c", smoke_code],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ) if install.returncode == 0 else None
            cli_result = subprocess.run(
                [str(environment / "bin" / cli[0]), *cli[1:]],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ) if install.returncode == 0 else None
            uninstall = subprocess.run(
                [str(python), "-m", "pip", "uninstall", "-y", *package_names],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            ) if install.returncode == 0 else None
            residue_code = (
                "import importlib.util; modules=" + repr(list(residue_modules))
                + "; raise SystemExit(1 if any(importlib.util.find_spec(x) for x in modules) else 0)"
            )
            residue = subprocess.run(
                [str(python), "-c", residue_code],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ) if uninstall and uninstall.returncode == 0 else None
            checks = (install, smoke, cli_result, uninstall, residue)
            results.append(
                {
                    "profile": index,
                    "passed": all(item is not None and item.returncode == 0 for item in checks),
                    "install_exit_code": install.returncode,
                    "smoke_exit_code": smoke.returncode if smoke else None,
                    "cli_exit_code": cli_result.returncode if cli_result else None,
                    "uninstall_exit_code": uninstall.returncode if uninstall else None,
                    "residue_exit_code": residue.returncode if residue else None,
                }
            )
    return results


def install_test(
    wheelhouse: Path,
    manifest: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    if manifest["violations"]:
        raise WheelhouseError("wheelhouse_manifest_has_violations")
    if manifest["missing_wheels"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "not_ready_missing_required_wheels",
            "exit_code": EXIT_NOT_READY,
            "missing_wheels": manifest["missing_wheels"],
            "venv_results": [],
            "real_wheelhouse_qualified": False,
            "execution": EXECUTION,
            "formal_validation_complete": False,
        }
    packages = [item["name"] for item in manifest["wheelhouse_index"]]
    results = _run_install_profiles(
        wheelhouse,
        manifest["install_plan"],
        packages,
        smoke_code=(
            "from fastapi import FastAPI; from fastapi.testclient import TestClient; "
            "import pydantic, rank_bm25, uvicorn; app=FastAPI(); "
            "app.get('/health')(lambda: {'ok': True}); "
            "assert TestClient(app).get('/health').json()=={'ok': True}; "
            "from scholar_agent.app.main import app as production_app; assert production_app.routes"
        ),
        cli=["uvicorn", "--help"],
        residue_modules=["fastapi", "httpx", "pydantic", "pytest", "rank_bm25", "uvicorn"],
        source_root=repository_root,
    )
    qualified = all(row["passed"] for row in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "wheelhouse_qualified" if qualified else "artifact_or_supply_chain_violation",
        "exit_code": EXIT_QUALIFIED if qualified else EXIT_VIOLATION,
        "missing_wheels": [],
        "venv_results": results,
        "real_wheelhouse_qualified": qualified,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.create_system = 3
    return info


def build_synthetic_wheel(
    output: Path,
    name: str,
    version: str,
    *,
    requires_dist: Sequence[str] = (),
    entry_point: bool = False,
) -> None:
    """Build a deterministic tiny wheel exclusively for temporary conformance tests."""

    normalized = name.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    module = normalized
    members: dict[str, bytes] = {
        f"{module}/__init__.py": f"__version__ = {version!r}\n".encode(),
        f"{module}/cli.py": b"def main():\n    return 0\n",
        f"{dist_info}/METADATA": (
            "\n".join(
                [
                    "Metadata-Version: 2.3",
                    f"Name: {name}",
                    f"Version: {version}",
                    "License: MIT",
                    *[f"Requires-Dist: {value}" for value in requires_dist],
                    "",
                ]
            ).encode()
        ),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: offline_wheelhouse_intake_v1\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if entry_point:
        members[f"{dist_info}/entry_points.txt"] = (
            f"[console_scripts]\nsynthetic-alpha = {module}.cli:main\n".encode()
        )
    rows = [
        (member, _record_hash(content), str(len(content)))
        for member, content in sorted(members.items())
    ]
    record_path = f"{dist_info}/RECORD"
    rows.append((record_path, "", ""))
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    members[record_path] = buffer.getvalue().encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for member, content in sorted(members.items()):
            archive.writestr(_zip_info(member), content)


def synthetic_install_test(protocol: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="synthetic-wheelhouse-") as temporary:
        wheelhouse = Path(temporary)
        build_synthetic_wheel(wheelhouse / "synthetic_beta-1.0-py3-none-any.whl", "synthetic-beta", "1.0")
        build_synthetic_wheel(
            wheelhouse / "synthetic_alpha-1.0-py3-none-any.whl",
            "synthetic-alpha",
            "1.0",
            requires_dist=["synthetic-beta==1.0"],
            entry_point=True,
        )
        lock = {
            "protocol": "python_dependency_lock_v1",
            "package_count": 2,
            "packages": [
                {"name": "synthetic-alpha", "version": "1.0", "groups": ["runtime"], "dependencies": ["synthetic-beta"]},
                {"name": "synthetic-beta", "version": "1.0", "groups": ["runtime"], "dependencies": []},
            ],
        }
        synthetic_protocol = json.loads(json.dumps(protocol))
        synthetic_protocol["dependency_lock"]["sha256"] = sha256_bytes(canonical_json(lock))
        manifest = build_manifest(wheelhouse, lock, synthetic_protocol)
        results = _run_install_profiles(
            wheelhouse,
            manifest["install_plan"],
            ["synthetic-alpha", "synthetic-beta"],
            smoke_code="import synthetic_alpha, synthetic_beta; assert synthetic_alpha.__version__ == '1.0'",
            cli=["synthetic-alpha", "--help"],
            residue_modules=["synthetic_alpha", "synthetic_beta"],
        )
    qualified = all(row["passed"] for row in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "synthetic_wheelhouse_qualified" if qualified else "artifact_or_supply_chain_violation",
        "exit_code": EXIT_QUALIFIED if qualified else EXIT_VIOLATION,
        "synthetic_only": True,
        "real_wheelhouse_qualified": False,
        "wheel_count": 2,
        "dependency_edge_count": 1,
        "venv_results": results,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def freeze_release_contract(
    repository_root: Path,
    protocol: Mapping[str, Any],
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind wheelhouse qualification to a new release contract without rewriting history."""

    specification = json.loads(
        (repository_root / "benchmark/release_candidate_reproducibility_v1_spec.json").read_text(
            encoding="utf-8"
        )
    )
    specification["source_commit"] = protocol["source_commit"]
    contract, closures = freeze_contract(repository_root, specification)
    frontend_protocol = json.loads(
        (repository_root / "benchmark/frontend_reproducible_build_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract["frontend_canonical_staging"] = frontend_protocol["canonical_staging"]
    contract["python_lock"] = {
        "path": protocol["release_python_closure_output"],
        "sha256": stable_digest(closures["python"]),
    }
    contract["python_dependency_lock"] = {
        "protocol": "python_dependency_lock_v1",
        "manifest_sha256": stable_digest(lock),
        "runtime_requires_dist": lock["runtime_requires_dist"],
        "lock_qualified": lock["lock_qualified"],
        "offline_install_qualified": manifest["real_wheelhouse_qualified"],
    }
    contract["offline_wheelhouse_intake"] = {
        "protocol": PROTOCOL,
        "manifest_path": protocol["manifest_output"],
        "manifest_sha256": sha256_bytes(canonical_json(manifest)),
        "expected_wheel_count": manifest["expected_wheel_count"],
        "accepted_wheel_count": manifest["accepted_wheel_count"],
        "real_wheelhouse_qualified": manifest["real_wheelhouse_qualified"],
    }
    return contract, closures["python"]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))
