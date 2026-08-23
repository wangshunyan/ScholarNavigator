from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from fetch_open_full_text import main


class _Response:
    status = 200
    headers = {"Content-Type": "text/plain; charset=utf-8"}

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):  # noqa: ANN201
        return self

    def __exit__(self, *args):  # noqa: ANN002, ANN201
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def _opener(response: _Response):
    def open_request(request, *, timeout: float):  # noqa: ANN001
        assert request.full_url == "https://open.example.test/paper"
        assert timeout == 2
        return response

    return open_request


def test_cli_emits_structured_success_and_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    exit_code = main(
        [
            "--url",
            "https://open.example.test/paper",
            "--license-id",
            "CC-BY-4.0",
            "--license-verified",
            "--allowed-host",
            "open.example.test",
            "--timeout-seconds",
            "2",
            "--output",
            str(output),
        ],
        opener=_opener(_Response(b"A licensed paragraph.")),
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["document"]["paragraphs"][0]["text"] == "A licensed paragraph."


def test_cli_requires_explicit_license_acknowledgement(tmp_path: Path) -> None:
    output = tmp_path / "blocked.json"
    exit_code = main(
        [
            "--url",
            "https://open.example.test/paper",
            "--license-id",
            "CC-BY-4.0",
            "--allowed-host",
            "open.example.test",
            "--output",
            str(output),
        ],
        opener=_opener(_Response(b"should not be fetched")),
    )
    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "license_unverified"
    assert payload["document"] is None
