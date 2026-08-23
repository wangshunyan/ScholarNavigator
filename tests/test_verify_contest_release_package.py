from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_contest_release_package import build_package
from scripts.verify_contest_release_package import ReleaseVerificationError, verify_package


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    return root


def test_verify_release_manifest_and_members(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    package = tmp_path / "release.zip"
    build_package(root, package)
    report = verify_package(package, expected_commit=__import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    assert report["status"] == "ready"
    assert report["file_count"] == 1
    assert report["network_requests"] == 0


def test_verify_rejects_tampered_member(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    package = tmp_path / "release.zip"
    build_package(root, package)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "src/main.py":
                data = b"changed\n"
            target.writestr(info, data)
    with pytest.raises(ReleaseVerificationError, match="member_hash_mismatch"):
        verify_package(tampered)
