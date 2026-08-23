from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.verify_contest_release_package import ReleaseVerificationError, verify_package


def test_verify_rejects_server_path_in_release_document(tmp_path: Path) -> None:
    package = tmp_path / "server-path.zip"
    data = b"server path: /mnt/highway1/private\n"
    manifest = {
        "schema_version": "contest-release-manifest-v1",
        "source_commit": "a" * 40,
        "file_count": 1,
        "files": [{"path": "docs/note.md", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("docs/note.md", data)
        archive.writestr("release-manifest.json", json.dumps(manifest))
    with pytest.raises(ReleaseVerificationError, match="forbidden_content_marker"):
        verify_package(package)
