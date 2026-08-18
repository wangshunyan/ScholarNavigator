from __future__ import annotations

import json
import base64
import csv
import hashlib
import io
import stat
import zipfile
from pathlib import Path

import pytest

from scholar_agent.evaluation.offline_wheelhouse_intake import (
    EXIT_NOT_READY,
    WheelhouseError,
    build_manifest,
    build_synthetic_wheel,
    freeze_release_contract,
    inspect_wheel,
    synthetic_install_test,
    verify_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _protocol() -> dict[str, object]:
    return _json("benchmark/offline_wheelhouse_intake_v1_protocol.json")


def _lock() -> dict[str, object]:
    return _json("benchmark/python_dependency_lock_v1_manifest.json")


def _rewrite(path: Path, changes: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members.update(changes)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)


def _rewrite_with_valid_record(path: Path, changes: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members.update(changes)
    record_path = next(name for name in members if name.endswith(".dist-info/RECORD"))
    members.pop(record_path)
    rows = []
    for name, content in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        rows.append((name, "sha256=" + digest.decode(), str(len(content))))
    rows.append((record_path, "", ""))
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    members[record_path] = buffer.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)


def _synthetic_lock() -> dict[str, object]:
    return {
        "protocol": "python_dependency_lock_v1",
        "package_count": 2,
        "packages": [
            {
                "name": "synthetic-alpha",
                "version": "1.0",
                "groups": ["runtime"],
                "dependencies": ["synthetic-beta"],
            },
            {
                "name": "synthetic-beta",
                "version": "1.0",
                "groups": ["runtime"],
                "dependencies": [],
            },
        ],
    }


def _synthetic_protocol(lock: dict[str, object]) -> dict[str, object]:
    from scholar_agent.evaluation.release_candidate_reproducibility import (
        canonical_json,
        sha256_bytes,
    )

    protocol = _protocol()
    protocol["dependency_lock"]["sha256"] = sha256_bytes(canonical_json(lock))
    return protocol


def _wheelhouse(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    lock = _synthetic_lock()
    protocol = _synthetic_protocol(lock)
    build_synthetic_wheel(
        tmp_path / "synthetic_beta-1.0-py3-none-any.whl",
        "synthetic-beta",
        "1.0",
    )
    build_synthetic_wheel(
        tmp_path / "synthetic_alpha-1.0-py3-none-any.whl",
        "synthetic-alpha",
        "1.0",
        requires_dist=["synthetic-beta==1.0"],
        entry_point=True,
    )
    return tmp_path, lock, protocol


def test_real_empty_wheelhouse_is_closed_and_deterministic(tmp_path: Path) -> None:
    first = build_manifest(tmp_path, _lock(), _protocol())
    second = build_manifest(tmp_path, _lock(), _protocol())
    assert first == second
    assert first["status"] == "not_ready_missing_required_wheels"
    assert first["exit_code"] == EXIT_NOT_READY
    assert first["expected_wheel_count"] == 23
    assert first["accepted_wheel_count"] == 0
    assert len(first["missing_wheels"]) == 23
    assert first["wheelhouse_index"] == []


def test_complete_synthetic_wheelhouse_reconstructs_lock_and_hash_plan(
    tmp_path: Path,
) -> None:
    wheelhouse, lock, protocol = _wheelhouse(tmp_path)
    manifest = build_manifest(wheelhouse, lock, protocol)
    assert manifest["status"] == "wheelhouse_qualified"
    assert manifest["dependency_edges"] == [
        {"from": "synthetic-alpha", "to": "synthetic-beta"}
    ]
    assert all(" --hash=sha256:" in row for row in manifest["install_plan"])
    assert all(
        row["source_type"] == "unknown" for row in manifest["wheelhouse_index"]
    )
    assert all(
        row["license_evidence"]["status"] == "declared_in_wheel_metadata"
        for row in manifest["wheelhouse_index"]
    )


@pytest.mark.parametrize(
    "member",
    ["../escape.py", "/absolute.py", "safe/../../escape.py"],
)
def test_path_traversal_and_absolute_members_are_rejected(
    tmp_path: Path, member: str
) -> None:
    wheel = tmp_path / "synthetic_alpha-1.0-py3-none-any.whl"
    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"bad")
    with pytest.raises(WheelhouseError, match="unsafe_wheel_member_path"):
        inspect_wheel(wheel, _protocol()["intake"])


def test_links_and_duplicate_zip_members_are_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "synthetic_alpha-1.0-py3-none-any.whl"
    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0")
    link = zipfile.ZipInfo("synthetic_alpha/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(link, b"target")
    with pytest.raises(WheelhouseError, match="wheel_link_member_rejected"):
        inspect_wheel(wheel, _protocol()["intake"])

    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0")
    with pytest.warns(UserWarning), zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("synthetic_alpha/__init__.py", b"duplicate")
    with pytest.raises(WheelhouseError, match="wheel_duplicate_member"):
        inspect_wheel(wheel, _protocol()["intake"])


def test_record_hash_tamper_and_metadata_filename_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "synthetic_alpha-1.0-py3-none-any.whl"
    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0")
    _rewrite(wheel, {"synthetic_alpha/__init__.py": b"tampered\n"})
    with pytest.raises(WheelhouseError, match="wheel_record_(hash|size)_invalid"):
        inspect_wheel(wheel, _protocol()["intake"])

    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0")
    metadata = (
        "Metadata-Version: 2.3\nName: confusing-name\nVersion: 1.0\n\n"
    ).encode()
    _rewrite(wheel, {"synthetic_alpha-1.0.dist-info/METADATA": metadata})
    with pytest.raises(WheelhouseError, match="wheel_name_metadata_mismatch"):
        inspect_wheel(wheel, _protocol()["intake"])


def test_wrong_tag_sdist_extra_and_duplicate_version_are_rejected(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "synthetic_alpha-1.0-cp27-cp27m-win32.whl"
    build_synthetic_wheel(wrong, "synthetic-alpha", "1.0")
    with pytest.raises(WheelhouseError, match="wheel_tag_incompatible"):
        inspect_wheel(wrong, _protocol()["intake"])

    wheelhouse, lock, protocol = _wheelhouse(tmp_path / "sdist")
    (wheelhouse / "synthetic-alpha-1.0.tar.gz").write_bytes(b"sdist")
    with pytest.raises(WheelhouseError, match="sdist_extra_or_directory_rejected"):
        build_manifest(wheelhouse, lock, protocol)

    wheelhouse, lock, protocol = _wheelhouse(tmp_path / "duplicate")
    original = wheelhouse / "synthetic_alpha-1.0-py3-none-any.whl"
    original.replace(wheelhouse / "synthetic_alpha-1.0-1-py3-none-any.whl")
    build_synthetic_wheel(
        wheelhouse / "synthetic_alpha-1.0-2-py3-none-any.whl",
        "synthetic-alpha",
        "1.0",
        requires_dist=["synthetic-beta==1.0"],
    )
    with pytest.raises(WheelhouseError, match="duplicate_locked_wheel"):
        build_manifest(wheelhouse, lock, protocol)


def test_dependency_closure_conflict_and_extra_wheel_are_rejected(
    tmp_path: Path,
) -> None:
    wheelhouse, lock, protocol = _wheelhouse(tmp_path / "conflict")
    alpha = wheelhouse / "synthetic_alpha-1.0-py3-none-any.whl"
    build_synthetic_wheel(
        alpha,
        "synthetic-alpha",
        "1.0",
        requires_dist=["synthetic-beta>=2"],
    )
    manifest = build_manifest(wheelhouse, lock, protocol)
    assert manifest["exit_code"] == 2
    assert "dependency_version_conflict:synthetic-alpha:synthetic-beta" in manifest[
        "violations"
    ]

    wheelhouse, lock, protocol = _wheelhouse(tmp_path / "extra")
    build_synthetic_wheel(
        wheelhouse / "unregistered-1.0-py3-none-any.whl",
        "unregistered",
        "1.0",
    )
    with pytest.raises(WheelhouseError, match="extra_or_version_conflicting_wheel"):
        build_manifest(wheelhouse, lock, protocol)


def test_cross_manifest_mix_and_missing_wheel_are_detected(tmp_path: Path) -> None:
    wheelhouse, lock, protocol = _wheelhouse(tmp_path)
    tracked = build_manifest(wheelhouse, lock, protocol)
    (wheelhouse / "synthetic_beta-1.0-py3-none-any.whl").unlink()
    report = verify_manifest(wheelhouse, lock, protocol, tracked)
    assert report["exit_code"] == 2
    assert "wheelhouse_manifest_drift" in report["violations"]


def test_compression_limits_and_entry_point_schema_are_enforced(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "synthetic_alpha-1.0-py3-none-any.whl"
    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0", entry_point=True)
    policy = dict(_protocol()["intake"])
    policy["max_total_uncompressed_bytes"] = 1
    with pytest.raises(WheelhouseError, match="wheel_total_size_limit"):
        inspect_wheel(wheel, policy)

    build_synthetic_wheel(wheel, "synthetic-alpha", "1.0", entry_point=True)
    _rewrite_with_valid_record(
        wheel,
        {
            "synthetic_alpha-1.0.dist-info/entry_points.txt": (
                b"[console_scripts]\nbad = os:system; rm -rf /\n"
            )
        },
    )
    with pytest.raises(WheelhouseError, match="entry_point_invalid"):
        inspect_wheel(wheel, _protocol()["intake"])


def test_synthetic_dual_venv_install_is_real_but_not_release_qualification() -> None:
    first = synthetic_install_test(_protocol())
    second = synthetic_install_test(_protocol())
    assert first == second
    assert first["status"] == "synthetic_wheelhouse_qualified"
    assert first["real_wheelhouse_qualified"] is False
    assert [row["passed"] for row in first["venv_results"]] == [True, True]


def test_release_contract_binds_real_wheelhouse_status() -> None:
    protocol = _protocol()
    lock = _lock()
    manifest = build_manifest(ROOT / "wheelhouse", lock, protocol)
    contract, closure = freeze_release_contract(ROOT, protocol, lock, manifest)
    intake = contract["offline_wheelhouse_intake"]
    assert intake["protocol"] == "offline_wheelhouse_intake_v1"
    assert intake["expected_wheel_count"] == 23
    assert intake["accepted_wheel_count"] == 0
    assert intake["real_wheelhouse_qualified"] is False
    assert contract["python_dependency_lock"]["offline_install_qualified"] is False
    assert closure["complete"] is True
