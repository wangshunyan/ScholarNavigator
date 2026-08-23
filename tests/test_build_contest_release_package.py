from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_contest_release_package import build_package


def test_release_package_excludes_sensitive_and_legacy_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "legacy" / "spar_original").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "datasets" / "semantic").mkdir(parents=True)
    (root / "datasets" / "local_bm25").mkdir(parents=True)
    (root / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=hidden\n", encoding="utf-8")
    (root / "legacy" / "spar_original" / "third_party.py").write_text("x\n", encoding="utf-8")
    (root / "outputs" / "run.json").write_text("{}\n", encoding="utf-8")
    (root / "datasets" / "semantic" / "model.bin").write_bytes(b"model")
    (root / "datasets" / "local_bm25" / "pasa_papers.jsonl").write_text(
        '{"_id":"x","arxiv_id":"2401.00001","title":"paper","abstract":"text"}\n',
        encoding="utf-8",
    )
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    output = tmp_path / "release.zip"
    report = build_package(root, output)
    assert report["file_count"] == 2
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "datasets/local_bm25/pasa_papers.jsonl",
            "src/main.py",
            "release-manifest.json",
        ]
        manifest = json.loads(archive.read("release-manifest.json"))
    assert manifest["file_count"] == 2
    assert {item["path"] for item in manifest["files"]} == {
        "datasets/local_bm25/pasa_papers.jsonl", "src/main.py"
    }
    assert report["manifest_name"] == "release-manifest.json"


def test_release_package_is_byte_stable_across_builds(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build_package(root, first)
    (root / "src" / "main.py").touch()
    build_package(root, second)
    assert first.read_bytes() == second.read_bytes()


def test_release_package_rejects_dirty_source_tree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "src").mkdir()
    source = root / "src" / "main.py"
    source.write_text("pass\n", encoding="utf-8")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="release_requires_clean_git_tree"):
        build_package(root, tmp_path / "dirty.zip")
