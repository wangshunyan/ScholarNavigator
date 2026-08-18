"""Small .env loader for local backend development.

This intentionally avoids adding a runtime dependency. Real environment
variables always win over values loaded from the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


ENV_FILE_ENV = "SCHOLAR_AGENT_ENV_FILE"
DEFAULT_ENV_FILE = ".env"
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_project_env(project_root: str | Path | None = None) -> bool:
    """Load the repository runtime environment using the application rule.

    A configured ``SCHOLAR_AGENT_ENV_FILE`` keeps its existing path semantics;
    otherwise the repository root is used instead of the process cwd. This
    makes CLI startup independent of the directory from which it is launched,
    while :func:`load_env_file` still keeps pre-existing environment variables
    authoritative.
    """

    configured_path = os.getenv(ENV_FILE_ENV)
    if configured_path:
        return load_env_file(configured_path)
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return load_env_file(root / DEFAULT_ENV_FILE)


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> bool:
    """Load KEY=VALUE pairs from a dotenv-style file if it exists.

    Returns True when a file was found and parsed. Missing files are ignored.
    """

    env_path = _resolve_env_path(path)
    if not env_path.exists() or not env_path.is_file():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return True


def _resolve_env_path(path: str | Path | None) -> Path:
    raw_path = path or os.getenv(ENV_FILE_ENV) or DEFAULT_ENV_FILE
    env_path = Path(raw_path).expanduser()
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    return env_path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").strip()
    if "=" not in stripped:
        return None

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not _KEY_PATTERN.fullmatch(key):
        return None

    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = _strip_inline_comment(value).strip()
    return key, value


def _strip_inline_comment(value: str) -> str:
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value
