"""Load and render versioned Markdown prompts from package resources."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import PurePosixPath
from typing import Any


PROMPT_PACKAGE = "scholar_agent.prompts"
MANIFEST_FILE = "manifest.json"
PAYLOAD_PLACEHOLDER = "{{payload}}"
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class PromptLoadError(RuntimeError):
    """Raised when a packaged prompt cannot be validated or rendered."""


@dataclass(frozen=True)
class PromptManifestEntry:
    """Validated metadata for one prompt in the manifest."""

    name: str
    version: str
    runtime_enabled: bool
    system_path: str | None = None
    user_path: str | None = None
    prompt_path: str | None = None


@dataclass(frozen=True)
class LoadedPrompt:
    """One validated active runtime prompt."""

    name: str
    version: str
    system_text: str
    user_text: str
    content_hash: str


def load_manifest() -> dict[str, PromptManifestEntry]:
    """Load and validate the package prompt manifest."""

    return _load_manifest(_resource_root())


def load_prompt(name: str) -> LoadedPrompt:
    """Load one active system/user prompt pair from package resources."""

    prompt_name = _normalize_name(name)
    root = _resource_root()
    manifest = _load_manifest(root)
    entry = manifest.get(prompt_name)
    if entry is None:
        raise PromptLoadError(f"Unknown prompt: {prompt_name}")
    if not entry.runtime_enabled:
        raise PromptLoadError(f"Prompt is not runtime-enabled: {prompt_name}")
    if entry.system_path is None or entry.user_path is None:
        raise PromptLoadError(f"Active prompt paths are incomplete: {prompt_name}")

    system_text = _read_nonempty_text(root, entry.system_path, prompt_name)
    user_text = _read_nonempty_text(root, entry.user_path, prompt_name)
    _validate_runtime_template(prompt_name, system_text, user_text)
    return LoadedPrompt(
        name=prompt_name,
        version=entry.version,
        system_text=system_text,
        user_text=user_text,
        content_hash=_content_hash(entry.version, system_text, user_text),
    )


def render_messages(name: str, payload: object) -> list[dict[str, str]]:
    """Render deterministic chat messages for one active prompt."""

    prompt = load_prompt(name)
    payload_text = _serialize_payload(name, payload)
    rendered_user = prompt.user_text.replace(PAYLOAD_PLACEHOLDER, payload_text)
    if PAYLOAD_PLACEHOLDER in rendered_user:
        raise PromptLoadError(f"Prompt payload rendering failed: {prompt.name}")
    return [
        {"role": "system", "content": prompt.system_text},
        {"role": "user", "content": rendered_user},
    ]


def render_untrusted_metadata_messages(
    name: str, payload: object
) -> list[dict[str, str]]:
    """Render one data-only metadata envelope without changing message roles."""

    prompt = load_prompt(name)
    envelope = {
        "boundary": {
            "contract": "untrusted_metadata_isolation_v1",
            "envelope_kind": "untrusted_academic_metadata_v1",
            "instruction_capability": False,
            "metadata_role": "untrusted_data",
        },
        "payload": payload,
    }
    payload_text = _serialize_payload(name, envelope, untrusted=True)
    rendered_user = prompt.user_text.replace(PAYLOAD_PLACEHOLDER, payload_text)
    messages = [
        {"role": "system", "content": prompt.system_text},
        {"role": "user", "content": rendered_user},
    ]
    validate_data_only_message_roles(messages)
    return messages


def validate_data_only_message_roles(messages: object) -> None:
    """Reject role/tool smuggling before an LLM client sees the messages."""

    if not isinstance(messages, list) or len(messages) != 2:
        raise PromptLoadError("Untrusted metadata messages must contain two roles")
    for index, expected_role in enumerate(("system", "user")):
        message = messages[index]
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise PromptLoadError("Untrusted metadata message fields are invalid")
        if message.get("role") != expected_role:
            raise PromptLoadError("Untrusted metadata message role is invalid")
        if not isinstance(message.get("content"), str):
            raise PromptLoadError("Untrusted metadata message content is invalid")


def _resource_root() -> Traversable:
    try:
        return resources.files(PROMPT_PACKAGE)
    except (ModuleNotFoundError, TypeError) as exc:
        raise PromptLoadError("Prompt package resources are unavailable") from exc


def _load_manifest(root: Traversable) -> dict[str, PromptManifestEntry]:
    manifest_text = _read_nonempty_text(root, MANIFEST_FILE, "manifest")
    try:
        raw_manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise PromptLoadError(
            f"Prompt manifest is invalid JSON at line {exc.lineno}"
        ) from None
    if not isinstance(raw_manifest, dict) or not raw_manifest:
        raise PromptLoadError("Prompt manifest must be a non-empty object")

    manifest: dict[str, PromptManifestEntry] = {}
    for raw_name, raw_entry in raw_manifest.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise PromptLoadError("Prompt manifest contains an invalid name")
        name = raw_name.strip()
        if not isinstance(raw_entry, dict):
            raise PromptLoadError(f"Prompt manifest entry must be an object: {name}")
        entry = _parse_manifest_entry(name, raw_entry)
        for path in _entry_paths(entry):
            _read_nonempty_text(root, path, name)
        manifest[name] = entry
    return manifest


def _parse_manifest_entry(
    name: str,
    raw_entry: dict[str, Any],
) -> PromptManifestEntry:
    allowed_fields = {
        "version",
        "runtime_enabled",
        "system",
        "user",
        "prompt",
    }
    unexpected = sorted(set(raw_entry) - allowed_fields)
    if unexpected:
        raise PromptLoadError(f"Prompt manifest has unsupported fields: {name}")

    version = raw_entry.get("version")
    runtime_enabled = raw_entry.get("runtime_enabled")
    if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
        raise PromptLoadError(f"Prompt version must use x.y.z format: {name}")
    if not isinstance(runtime_enabled, bool):
        raise PromptLoadError(f"Prompt runtime_enabled must be boolean: {name}")

    system_path = _optional_relative_path(raw_entry.get("system"), name, "system")
    user_path = _optional_relative_path(raw_entry.get("user"), name, "user")
    prompt_path = _optional_relative_path(raw_entry.get("prompt"), name, "prompt")
    if runtime_enabled:
        if system_path is None or user_path is None or prompt_path is not None:
            raise PromptLoadError(
                f"Active prompt must define only system and user files: {name}"
            )
    elif prompt_path is None or system_path is not None or user_path is not None:
        raise PromptLoadError(
            f"Inactive prompt must define only one prompt file: {name}"
        )
    return PromptManifestEntry(
        name=name,
        version=version,
        runtime_enabled=runtime_enabled,
        system_path=system_path,
        user_path=user_path,
        prompt_path=prompt_path,
    )


def _optional_relative_path(value: Any, name: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PromptLoadError(f"Prompt {field} path is invalid: {name}")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
        raise PromptLoadError(f"Prompt {field} path must be relative Markdown: {name}")
    return path.as_posix()


def _entry_paths(entry: PromptManifestEntry) -> tuple[str, ...]:
    return tuple(
        path
        for path in (entry.system_path, entry.user_path, entry.prompt_path)
        if path is not None
    )


def _read_nonempty_text(
    root: Traversable,
    relative_path: str,
    prompt_name: str,
) -> str:
    resource = root.joinpath(relative_path)
    try:
        if not resource.is_file():
            raise PromptLoadError(f"Prompt file is missing: {prompt_name}/{relative_path}")
        text = resource.read_text(encoding="utf-8")
    except PromptLoadError:
        raise
    except (OSError, UnicodeError):
        raise PromptLoadError(
            f"Prompt file cannot be read as UTF-8: {prompt_name}/{relative_path}"
        ) from None
    normalized = text.strip()
    if not normalized:
        raise PromptLoadError(f"Prompt file is empty: {prompt_name}/{relative_path}")
    return normalized


def _validate_runtime_template(
    name: str,
    system_text: str,
    user_text: str,
) -> None:
    if PAYLOAD_PLACEHOLDER in system_text:
        raise PromptLoadError(f"Prompt payload placeholder is in system text: {name}")
    if user_text.count(PAYLOAD_PLACEHOLDER) != 1:
        raise PromptLoadError(
            f"Prompt user template must contain one payload placeholder: {name}"
        )


def _serialize_payload(
    name: str, payload: object, *, untrusted: bool = False
) -> str:
    serializable = (
        payload.model_dump(mode="json")
        if hasattr(payload, "model_dump")
        else payload
    )
    try:
        serialized = json.dumps(
            serializable,
            ensure_ascii=untrusted,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        if untrusted:
            # Keep markup-looking delimiters inside JSON string data instead of
            # allowing them to resemble prompt/template boundaries.
            serialized = (
                serialized.replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
            )
        return serialized
    except (TypeError, ValueError):
        raise PromptLoadError(
            f"Prompt payload is not JSON-serializable: {_normalize_name(name)}"
        ) from None


def _content_hash(version: str, system_text: str, user_text: str) -> str:
    content = "\0".join((version, system_text, user_text)).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _normalize_name(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        raise PromptLoadError("Prompt name must not be empty")
    return normalized
