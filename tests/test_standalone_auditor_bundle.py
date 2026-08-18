from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_standalone_auditor_bundle.py"
SPEC = importlib.util.spec_from_file_location("standalone_bundle_cli", SCRIPT)
assert SPEC and SPEC.loader
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def _build(tmp_path: Path, name: str = "bundle.zip") -> Path:
    target = tmp_path / name
    report = bundle.build_archive(
        ROOT / "benchmark" / "standalone_auditor_bundle_v1_contract.json",
        target,
        ROOT,
    )
    assert report["status"] == "verified_with_declared_blockers"
    return target


def _rewrite(source: Path, target: Path, transform) -> None:
    with zipfile.ZipFile(source) as archive:
        rows = [(info, archive.read(info)) for info in archive.infolist()]
    rows = transform(rows)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for old, content in rows:
            info = zipfile.ZipInfo(old.filename, old.date_time)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = old.external_attr
            info.create_system = old.create_system
            archive.writestr(info, content)


def _reseal(source: Path, target: Path, member: str, content: bytes) -> None:
    with zipfile.ZipFile(source) as archive:
        rows = {info.filename: archive.read(info) for info in archive.infolist()}
    rows[member] = content
    manifest = json.loads(rows["manifest.json"])
    for item in manifest["files"]:
        if item["path"] == member:
            item["size"] = len(content)
            item["sha256"] = bundle.digest(content)
            break
    else:
        raise AssertionError(member)
    manifest["manifest_self_sha256"] = bundle._self_hash(manifest)
    rows["manifest.json"] = bundle.canonical_bytes(manifest)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(rows.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, value)


def _run_trusted_verifier(tmp_path: Path, archive: Path, *extra: str) -> subprocess.CompletedProcess[bytes]:
    verifier = tmp_path / "trusted" / "verify.py"
    verifier.parent.mkdir(exist_ok=True)
    verifier.write_bytes(SCRIPT.read_bytes())
    return subprocess.run(
        [sys.executable, "-I", "-S", str(verifier), "verify", str(archive), *extra],
        cwd=tmp_path,
        env={"HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path / "tmp"), "PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "11"},
        capture_output=True,
        check=False,
    )


def test_two_builds_and_isolated_standard_library_verify_are_byte_stable(tmp_path: Path) -> None:
    first = _build(tmp_path, "one.zip")
    second = _build(tmp_path, "two.zip")
    assert first.read_bytes() == second.read_bytes()
    verifier = tmp_path / "trusted-verifier.py"
    verifier.write_bytes(SCRIPT.read_bytes())
    reports = []
    for suffix in ("a", "b"):
        cwd = tmp_path / suffix / "cwd"
        home = tmp_path / suffix / "home"
        scratch = tmp_path / suffix / "tmp"
        cwd.mkdir(parents=True); home.mkdir(); scratch.mkdir()
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(verifier), "verify", str(first)],
            cwd=cwd,
            env={"HOME": str(home), "TMPDIR": str(scratch), "PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "random" if suffix == "a" else "7"},
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert completed.stderr == b""
        reports.append(completed.stdout)
    assert reports[0] == reports[1]
    assert json.loads(reports[0])["formal_validation_complete"] is False


@pytest.mark.parametrize("kind", ["tamper", "missing", "extra"])
def test_inventory_tamper_missing_and_extra_are_rejected(tmp_path: Path, kind: str) -> None:
    source = _build(tmp_path)
    target = tmp_path / f"{kind}.zip"
    def change(rows):
        if kind == "tamper":
            return [(info, content + b" ") if info.filename == "claims.json" else (info, content) for info, content in rows]
        if kind == "missing":
            return [(info, content) for info, content in rows if info.filename != "claims.json"]
        info = zipfile.ZipInfo("extra.json", (1980, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16; info.create_system = 3
        return [*rows, (info, b"{}\n")]
    _rewrite(source, target, change)
    with pytest.raises(bundle.AuditError):
        bundle.verify_archive(target)


def test_hash_closed_legacy_archive_without_revocation_state_is_rejected(
    tmp_path: Path,
) -> None:
    source = _build(tmp_path)
    legacy = tmp_path / "legacy.zip"
    with zipfile.ZipFile(source) as archive:
        rows = {info.filename: archive.read(info) for info in archive.infolist()}
    rows.pop("revocation.json")
    manifest = json.loads(rows["manifest.json"])
    manifest["files"] = [
        item for item in manifest["files"] if item["path"] != "revocation.json"
    ]
    manifest["manifest_self_sha256"] = bundle._self_hash(manifest)
    rows["manifest.json"] = bundle.canonical_bytes(manifest)
    with zipfile.ZipFile(legacy, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(rows.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, content)
    with pytest.raises(bundle.AuditError, match="archive_inventory_not_closed"):
        bundle.verify_archive(legacy)


def _mutate_json(source: Path, target: Path, member: str, mutate) -> None:
    with zipfile.ZipFile(source) as archive:
        value = json.loads(archive.read(member))
    mutate(value)
    _reseal(source, target, member, bundle.canonical_bytes(value))


@pytest.mark.parametrize(
    ("member", "mutation"),
    [
        ("blockers.json", lambda value: value["blockers"].pop()),
        ("readiness.json", lambda value: value.__setitem__("formal_validation_complete", True)),
        ("freshness.json", lambda value: value.__setitem__("stale_count", 1)),
        ("policy.json", lambda value: value.__setitem__("deterministic_tiebreak_v2_default_enabled", True)),
        ("preregistration.json", lambda value: value.__setitem__("state", "invalid_post_evidence_change")),
        ("evidence_index.json", lambda value: value["evidence"][0].__setitem__("verification_scope", "externally_verified")),
        ("protocol_dependencies.json", lambda value: value["implementation_commit_ancestry"].__setitem__("head_commit", "0" * 40)),
    ],
)
def test_claim_blocker_freshness_policy_and_commit_attacks_fail(
    tmp_path: Path, member: str, mutation
) -> None:
    source = _build(tmp_path)
    target = tmp_path / "attack.zip"
    _mutate_json(source, target, member, mutation)
    with pytest.raises(bundle.AuditError):
        bundle.verify_archive(target)


def test_resealed_malformed_claims_and_duplicate_manifest_inventory_fail_closed(tmp_path: Path) -> None:
    source = _build(tmp_path)
    malformed = tmp_path / "malformed.zip"
    _reseal(source, malformed, "claims.json", bundle.canonical_bytes({"claims": "invalid", "protocol": bundle.PROTOCOL, "schema_version": "1"}))
    completed = _run_trusted_verifier(tmp_path, malformed)
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["error_code"] == "claims_schema_invalid"

    with zipfile.ZipFile(source) as archive:
        rows = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(rows["manifest.json"])
    manifest["files"].append(dict(manifest["files"][0]))
    manifest["manifest_self_sha256"] = bundle._self_hash(manifest)
    rows["manifest.json"] = bundle.canonical_bytes(manifest)
    duplicate = tmp_path / "duplicate-inventory.zip"
    with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(rows.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16; info.create_system = 3
            archive.writestr(info, content)
    completed = _run_trusted_verifier(tmp_path, duplicate)
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["error_code"] == "duplicate_manifest_inventory_entry"


def test_duplicate_key_path_traversal_and_link_are_rejected(tmp_path: Path) -> None:
    source = _build(tmp_path)
    duplicate = tmp_path / "duplicate-key.zip"
    def duplicate_json(rows):
        return [(info, b'{"schema_version":"1","schema_version":"1"}\n') if info.filename == "claims.json" else (info, content) for info, content in rows]
    _rewrite(source, duplicate, duplicate_json)
    with pytest.raises(bundle.AuditError): bundle.verify_archive(duplicate)

    for name, mode in (("../escape", 0o100644), ("C:/absolute", 0o100644), ("link", 0o120777)):
        target = tmp_path / ("path.zip" if ".." in name else "link.zip")
        def add(rows, name=name, mode=mode):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.external_attr = mode << 16; info.create_system = 3
            return [*rows, (info, b"x")]
        _rewrite(source, target, add)
        with pytest.raises(bundle.AuditError): bundle.verify_archive(target)


def test_malformed_utf8_nonfinite_and_resource_limits_are_rejected(tmp_path: Path) -> None:
    source = _build(tmp_path)
    for suffix, content in (("utf8", b"\xff"), ("nan", b'{"value":NaN}\n')):
        target = tmp_path / f"{suffix}.zip"
        def change(rows, content=content):
            return [(info, content) if info.filename == "claims.json" else (info, value) for info, value in rows]
        _rewrite(source, target, change)
        with pytest.raises(bundle.AuditError): bundle.verify_archive(target)

    target = tmp_path / "oversized.zip"
    def oversized(rows):
        info = zipfile.ZipInfo("large.bin", (1980, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16; info.create_system = 3
        return [*rows, (info, b"x" * (1024 * 1024 + 1))]
    _rewrite(source, target, oversized)
    with pytest.raises(bundle.AuditError, match="resource_limit"):
        bundle.verify_archive(target)


def test_recursive_and_resource_boundaries_have_stable_cli_errors(tmp_path: Path) -> None:
    source = _build(tmp_path)
    recursive = tmp_path / "recursive.zip"
    deep = ("[" * 2000 + "0" + "]" * 2000 + "\n").encode()
    _reseal(source, recursive, "claims.json", deep)
    completed = _run_trusted_verifier(tmp_path, recursive)
    assert completed.returncode == 2
    assert completed.stderr == b""
    assert json.loads(completed.stdout)["status"] == "integrity_or_claim_violation"

    oversized = tmp_path / "resource.zip"
    def add_large(rows):
        info = zipfile.ZipInfo("large.bin", (1980, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16; info.create_system = 3
        return [*rows, (info, b"x" * (1024 * 1024 + 1))]
    _rewrite(source, oversized, add_large)
    completed = _run_trusted_verifier(tmp_path, oversized)
    assert completed.returncode == 2
    assert completed.stderr == b""


def test_claim_reference_and_overclaim_are_rejected_after_resealing(tmp_path: Path) -> None:
    source = _build(tmp_path)
    with zipfile.ZipFile(source) as archive:
        rows = {info.filename: archive.read(info) for info in archive.infolist()}
    claims = json.loads(rows["claims.json"]); claims["claims"][0]["evidence_ids"] = ["unknown"]
    rows["claims.json"] = bundle.canonical_bytes(claims)
    manifest = json.loads(rows["manifest.json"])
    for item in manifest["files"]:
        if item["path"] == "claims.json": item["size"] = len(rows["claims.json"]); item["sha256"] = bundle.digest(rows["claims.json"])
    manifest["manifest_self_sha256"] = bundle._self_hash(manifest)
    rows["manifest.json"] = bundle.canonical_bytes(manifest)
    target = tmp_path / "resealed.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(rows.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.external_attr = 0o100644 << 16; info.create_system = 3
            archive.writestr(info, content)
    with pytest.raises(bundle.AuditError, match="claim_evidence_reference_missing"):
        bundle.verify_archive(target)


def _audit_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    contract = json.loads((ROOT / "benchmark/standalone_auditor_bundle_v1_contract.json").read_text())
    contract_path = root / "benchmark/standalone_auditor_bundle_v1_contract.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(bundle.canonical_bytes(contract))
    for relative in contract["sources"].values():
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # The copied readiness contract binds the copied standalone contract.
    readiness_contract = json.loads((root / contract["sources"]["readiness_contract"]).read_text())
    for item in readiness_contract["evidence"]:
        if item.get("path") == "benchmark/standalone_auditor_bundle_v1_contract.json":
            item["sha256"] = bundle.digest(contract_path.read_bytes())
    (root / contract["sources"]["readiness_contract"]).write_bytes(bundle.canonical_bytes(readiness_contract))
    return root, contract_path


def test_audit_readiness_distinguishes_missing_and_drifted_sources(tmp_path: Path) -> None:
    root, contract_path = _audit_root(tmp_path)
    contract = json.loads(contract_path.read_text())
    missing = root / contract["sources"]["claims"]
    missing.rename(missing.with_suffix(".absent"))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "audit-readiness", "--repository-root", str(root), "--contract", str(contract_path)],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 3
    assert completed.stderr == b""

    root, contract_path = _audit_root(tmp_path / "drift")
    contract = json.loads(contract_path.read_text())
    claims_path = root / contract["sources"]["claims"]
    claims = json.loads(claims_path.read_text())
    claims["claim_count"] += 1
    claims_path.write_bytes(bundle.canonical_bytes(claims))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "audit-readiness", "--repository-root", str(root), "--contract", str(contract_path)],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stderr == b""
