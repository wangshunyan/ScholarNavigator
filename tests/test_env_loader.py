from __future__ import annotations

import os
from pathlib import Path

from scholar_agent.core import env_loader
from scholar_agent.core.env_loader import load_env_file, load_project_env
from scholar_agent.llm.provider import get_llm_request_options, get_llm_runtime_config


ENV_KEYS = (
    "SCHOLAR_AGENT_LLM_PROVIDER",
    "SCHOLAR_AGENT_LLM_MODEL",
    "SCHOLAR_AGENT_LLM_API_KEY",
    "SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT",
    "SCHOLAR_AGENT_LLM_BASE_URL",
    "OPENALEX_MAILTO",
)


def test_load_env_file_sets_values_without_overriding_existing_env(
    tmp_path,
    monkeypatch,
) -> None:
    original_env = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    try:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "\n".join(
                [
                    "SCHOLAR_AGENT_LLM_PROVIDER=openai_compatible",
                    "SCHOLAR_AGENT_LLM_MODEL=gpt-4.1-mini",
                    "SCHOLAR_AGENT_LLM_API_KEY=file-key",
                    "SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT=1 # enable judgement",
                    'SCHOLAR_AGENT_LLM_BASE_URL="https://api.example.test/v1"',
                    "export OPENALEX_MAILTO=team@example.test",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("SCHOLAR_AGENT_LLM_API_KEY", "shell-key")

        loaded = load_env_file(env_file)

        assert loaded is True
        assert os.environ["SCHOLAR_AGENT_LLM_PROVIDER"] == "openai_compatible"
        assert os.environ["SCHOLAR_AGENT_LLM_MODEL"] == "gpt-4.1-mini"
        assert os.environ["SCHOLAR_AGENT_LLM_API_KEY"] == "shell-key"
        assert os.environ["SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT"] == "1"
        assert os.environ["SCHOLAR_AGENT_LLM_BASE_URL"] == (
            "https://api.example.test/v1"
        )
        assert os.environ["OPENALEX_MAILTO"] == "team@example.test"
    finally:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
            if original_env[key] is not None:
                os.environ[key] = original_env[key]


def test_load_env_file_can_override_when_requested(tmp_path, monkeypatch) -> None:
    original_api_key = os.environ.get("SCHOLAR_AGENT_LLM_API_KEY")
    try:
        env_file = tmp_path / ".env"
        env_file.write_text("SCHOLAR_AGENT_LLM_API_KEY=file-key\n", encoding="utf-8")
        monkeypatch.setenv("SCHOLAR_AGENT_LLM_API_KEY", "shell-key")

        load_env_file(env_file, override=True)

        assert os.environ["SCHOLAR_AGENT_LLM_API_KEY"] == "file-key"
    finally:
        os.environ.pop("SCHOLAR_AGENT_LLM_API_KEY", None)
        if original_api_key is not None:
            os.environ["SCHOLAR_AGENT_LLM_API_KEY"] = original_api_key


def test_load_env_file_ignores_missing_file(tmp_path) -> None:
    assert load_env_file(tmp_path / "missing.env") is False


def test_load_project_env_uses_repository_root_from_other_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "project"
    repo_root.mkdir()
    (repo_root / ".env").write_text(
        "SCHOLAR_AGENT_LLM_PROVIDER=openai_compatible\n"
        "SCHOLAR_AGENT_LLM_MODEL=fixture-model\n"
        "SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS=17\n",
        encoding="utf-8",
    )
    keys = (
        "SCHOLAR_AGENT_LLM_PROVIDER",
        "SCHOLAR_AGENT_LLM_MODEL",
        "SCHOLAR_AGENT_LLM_BASE_URL",
        "SCHOLAR_AGENT_LLM_API_KEY",
        "SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS",
        "SCHOLAR_AGENT_ENV_FILE",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")

    assert load_project_env(repo_root) is True
    config = get_llm_runtime_config()
    assert (config.provider, config.model, config.available, config.reason) == (
        "openai_compatible",
        "fixture-model",
        False,
        "missing_env:SCHOLAR_AGENT_LLM_BASE_URL,SCHOLAR_AGENT_LLM_API_KEY",
    )
    assert get_llm_request_options()["timeout_seconds"] == 17.0
    for key in (
        "SCHOLAR_AGENT_LLM_PROVIDER",
        "SCHOLAR_AGENT_LLM_MODEL",
        "SCHOLAR_AGENT_LLM_BASE_URL",
        "SCHOLAR_AGENT_LLM_API_KEY",
        "SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


def test_load_project_env_infers_repository_root_without_argument(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "inferred-project"
    fake_module = repo_root / "src" / "scholar_agent" / "core" / "env_loader.py"
    fake_module.parent.mkdir(parents=True)
    (repo_root / ".env").write_text(
        "SCHOLAR_AGENT_LLM_PROVIDER=disabled\n"
        "SCHOLAR_AGENT_LLM_MODEL=inferred-model\n",
        encoding="utf-8",
    )
    for key in (
        "SCHOLAR_AGENT_LLM_PROVIDER",
        "SCHOLAR_AGENT_LLM_MODEL",
        "SCHOLAR_AGENT_ENV_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_loader, "__file__", str(fake_module))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_project_env() is True
    config = get_llm_runtime_config()
    assert config.provider == "disabled"
    assert config.model == "inferred-model"
    monkeypatch.delenv("SCHOLAR_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SCHOLAR_AGENT_LLM_MODEL", raising=False)


def test_load_project_env_preserves_explicit_environment_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SCHOLAR_AGENT_LLM_PROVIDER=disabled\n"
        "SCHOLAR_AGENT_LLM_MODEL=file-model\n"
        "SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS=11\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHOLAR_AGENT_ENV_FILE", str(env_file))
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_MODEL", "shell-model")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS", "23")

    assert load_project_env(tmp_path / "unused-project") is True
    config = get_llm_runtime_config()
    assert config.provider == "openai_compatible"
    assert config.model == "shell-model"
    assert get_llm_request_options()["timeout_seconds"] == 23.0


def test_missing_runtime_configuration_remains_explicit(monkeypatch) -> None:
    for key in (
        "SCHOLAR_AGENT_LLM_PROVIDER",
        "SCHOLAR_AGENT_LLM_MODEL",
        "SCHOLAR_AGENT_LLM_BASE_URL",
        "SCHOLAR_AGENT_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    config = get_llm_runtime_config()
    assert config.available is False
    assert config.reason == "provider_disabled"
