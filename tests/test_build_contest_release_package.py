from __future__ import annotations

import zipfile
from pathlib import Path

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
    output = tmp_path / "release.zip"
    report = build_package(root, output)
    assert report["file_count"] == 2
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "datasets/local_bm25/pasa_papers.jsonl",
            "src/main.py",
        ]
