#!/usr/bin/env python3
"""Generate and verify the offline Python dependency lock qualification."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.python_dependency_lock import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    DependencyLockError,
    DependencyLockNotReady,
    build_manifest,
    freeze_release_contract,
    lock_text,
    offline_install,
    verify_manifest,
    verify_wheel_metadata,
    write_json,
)
from scholar_agent.evaluation.release_candidate_reproducibility import (  # noqa: E402
    EXECUTION,
    build_python_lock,
    build_wheel,
    canonical_json,
    materialize_source,
    stable_digest,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/python_dependency_lock_v1_protocol.json"
DEFAULT_MANIFEST = ROOT / "benchmark/python_dependency_lock_v1_manifest.json"
DEFAULT_RELEASE_CONTRACT = ROOT / "benchmark/python_dependency_lock_v1_release_contract.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "verify", "offline-install", "audit-release"):
        command = commands.add_parser(name)
        command.add_argument("--report")
    return parser


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(status: str, code: int, reason: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": "python_dependency_lock_v1",
        "status": status,
        "exit_code": code,
        "reason": reason,
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def _generate(protocol: dict[str, object], manifest_path: Path) -> dict[str, object]:
    manifest = build_manifest(ROOT, protocol)
    write_json(manifest_path, manifest)
    for group, relative in protocol["lock_outputs"].items():
        (ROOT / relative).write_bytes(lock_text(manifest, group))
    contract, closure = freeze_release_contract(ROOT, protocol, manifest)
    write_json(DEFAULT_RELEASE_CONTRACT, contract)
    write_json(ROOT / contract["python_lock"]["path"], closure)
    return verify_manifest(ROOT, protocol, manifest)


def _audit_release(
    protocol: dict[str, object], manifest: dict[str, object]
) -> dict[str, object]:
    verification = verify_manifest(ROOT, protocol, manifest)
    contract = _load(DEFAULT_RELEASE_CONTRACT)
    release_lock = build_python_lock(ROOT / "requirements.txt")
    expected_runtime = sorted(
        (item["name"], item["version"])
        for item in manifest["packages"]
        if "runtime" in item["groups"]
    )
    actual_runtime = sorted(
        (item["name"], item["version"]) for item in release_lock["packages"]
    )
    with tempfile.TemporaryDirectory(prefix="python-lock-wheel-") as temporary:
        root = Path(temporary)
        source = root / "source"
        materialize_source(ROOT, contract, source)
        wheel = root / "spar_scholar_agent-0.1.0-py3-none-any.whl"
        build_wheel(source, wheel, contract)
        wheel_report = verify_wheel_metadata(wheel, manifest)
    violations = list(verification["violations"])
    violations.extend(wheel_report["violations"])
    if actual_runtime != expected_runtime:
        violations.append("release_sbom_runtime_closure_mismatch")
    if violations:
        status, code = "lock_or_metadata_violation", EXIT_VIOLATION
    elif not manifest["offline_install_qualified"]:
        status, code = "not_ready_missing_verified_version_or_artifact", EXIT_NOT_READY
    else:
        status, code = "dependency_lock_qualified", 0
    return {
        "schema_version": "1",
        "protocol": "python_dependency_lock_v1",
        "status": status,
        "exit_code": code,
        "lock_verification": verification,
        "wheel_metadata": wheel_report,
        "release_sbom_consistency": {
            "passed": actual_runtime == expected_runtime,
            "runtime_package_count": len(actual_runtime),
            "unknown_license_count": sum(
                item["license"] == "unknown" for item in release_lock["packages"]
            ),
        },
        "release_contract_sha256": stable_digest(contract),
        "violations": sorted(set(violations)),
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }


def main(argv: list[str] | None = None) -> int:
    report: dict[str, object]
    try:
        args = _parser().parse_args(argv)
        protocol = _load(Path(args.protocol))
        manifest_path = Path(args.manifest)
        if args.command == "generate":
            report = _generate(protocol, manifest_path)
        else:
            manifest = _load(manifest_path)
            if args.command == "verify":
                report = verify_manifest(ROOT, protocol, manifest)
            elif args.command == "offline-install":
                contract = _load(DEFAULT_RELEASE_CONTRACT)
                report = offline_install(ROOT, protocol, manifest, contract)
            else:
                report = _audit_release(protocol, manifest)
        if args.report:
            write_json(Path(args.report), report)
    except DependencyLockNotReady as exc:
        report = _error("not_ready_missing_verified_version_or_artifact", EXIT_NOT_READY, str(exc))
    except DependencyLockError as exc:
        report = _error("lock_or_metadata_violation", EXIT_VIOLATION, str(exc))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        report = _error("lock_or_metadata_violation", EXIT_VIOLATION, "controlled_input_or_io_failure")
    except SystemExit:
        report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
