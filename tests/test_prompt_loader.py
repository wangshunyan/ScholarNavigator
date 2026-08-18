from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.prompts import loader
from scholar_agent.prompts.loader import (
    PAYLOAD_PLACEHOLDER,
    PromptLoadError,
    load_manifest,
    load_prompt,
    render_messages,
)


def test_manifest_is_readable_and_has_expected_statuses() -> None:
    manifest = load_manifest()

    assert manifest["query_understanding"].runtime_enabled is True
    assert manifest["relevance_judgement"].runtime_enabled is True
    assert manifest["llm_query_planning"].runtime_enabled is True
    assert manifest["llm_constrained_rewrite"].runtime_enabled is True
    assert manifest["llm_relevance_judge_v1_1"].runtime_enabled is True
    assert manifest["llm_relevance_adjudicator_v1_1"].runtime_enabled is True
    assert manifest["judge_backend_qualification_v1"].runtime_enabled is True
    assert manifest["query_evolution"].runtime_enabled is False
    assert manifest["reranking"].runtime_enabled is False
    assert manifest["synthesis"].runtime_enabled is False


@pytest.mark.parametrize(
    "name",
    [
        "query_understanding",
        "relevance_judgement",
        "llm_query_planning",
        "llm_constrained_rewrite",
    ],
)
def test_active_prompt_loads_with_version_and_hash(name: str) -> None:
    prompt = load_prompt(name)

    assert prompt.name == name
    assert prompt.version == "1.0.0"
    assert prompt.system_text
    assert prompt.user_text
    assert len(prompt.content_hash) == 64
    assert prompt.content_hash == load_prompt(name).content_hash


@pytest.mark.parametrize(
    "name",
    ["llm_relevance_judge_v1_1", "llm_relevance_adjudicator_v1_1"],
)
def test_llm_relevance_v1_1_prompts_load_with_frozen_version(name: str) -> None:
    prompt = load_prompt(name)

    assert prompt.version == "1.1.0"
    assert "{{payload}}" in prompt.user_text
    assert len(prompt.content_hash) == 64


def test_judge_backend_qualification_prompt_loads_with_frozen_version() -> None:
    prompt = load_prompt("judge_backend_qualification_v1")

    assert prompt.version == "1.0.0"
    assert "{{payload}}" in prompt.user_text
    assert "exactly one JSON object" in prompt.system_text
    assert "not judging research relevance" in prompt.system_text


@pytest.mark.parametrize(
    "name",
    ["query_evolution", "reranking", "synthesis"],
)
def test_inactive_prompt_cannot_be_loaded_as_runtime_prompt(name: str) -> None:
    with pytest.raises(PromptLoadError, match="not runtime-enabled"):
        load_prompt(name)


def test_missing_prompt_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_prompt_tree(tmp_path, system_text=None)
    monkeypatch.setattr(loader, "_resource_root", lambda: tmp_path)

    with pytest.raises(PromptLoadError, match="file is missing"):
        load_prompt("query_understanding")


def test_empty_prompt_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_prompt_tree(tmp_path, system_text=" \n")
    monkeypatch.setattr(loader, "_resource_root", lambda: tmp_path)

    with pytest.raises(PromptLoadError, match="file is empty"):
        load_prompt("query_understanding")


def test_payload_is_replaced_and_chinese_json_remains_readable() -> None:
    messages = render_messages(
        "query_understanding",
        {"查询": "近三年论文", "数量": 5},
    )

    assert PAYLOAD_PLACEHOLDER not in messages[1]["content"]
    assert '"查询": "近三年论文"' in messages[1]["content"]
    assert "\\u67e5" not in messages[1]["content"]


def test_same_payload_generates_identical_messages() -> None:
    payload = {"query": "LLM reranking", "options": {"top_k": 5}}

    assert render_messages("query_understanding", payload) == render_messages(
        "query_understanding",
        payload,
    )


def test_loading_does_not_depend_on_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert load_prompt("query_understanding").version == "1.0.0"


def test_prompt_files_do_not_contain_api_key_text() -> None:
    root = loader._resource_root()
    for entry in load_manifest().values():
        paths = (entry.system_path, entry.user_path, entry.prompt_path)
        for path in (value for value in paths if value is not None):
            text = root.joinpath(path).read_text(encoding="utf-8").casefold()
            assert "api key" not in text
            assert "api_key" not in text


def test_query_understanding_prompt_matches_parser_fields() -> None:
    text = load_prompt("query_understanding").system_text

    for field in (
        "language",
        "intent",
        "domain",
        "constraints",
        "subqueries",
        "selected_sources",
        "warnings",
    ):
        assert field in text
    for source in ("openalex", "arxiv", "semantic_scholar", "pubmed"):
        assert source in text


def test_judgement_prompt_uses_batch_schema_and_evidence_sources() -> None:
    text = load_prompt("relevance_judgement").system_text

    assert '"judgements"' in text
    assert '"paper_index"' in text
    assert "insufficient_evidence" in text
    for source in ("title", "abstract", "venue", "metadata"):
        assert source in text


def test_llm_query_planning_prompt_has_strict_bounded_schema() -> None:
    text = load_prompt("llm_query_planning").system_text

    assert "supplemental_queries" in text
    assert "最多两条" in text
    assert "不得猜测具体论文标题" in text
    assert "DOI" in text


def test_llm_constrained_rewrite_prompt_has_strict_safety_contract() -> None:
    text = load_prompt("llm_constrained_rewrite").system_text

    assert '"rewritten_query"' in text
    assert "只生成一条" in text
    assert "protected_terms" in text
    assert "不得猜测或生成论文标题" in text
    assert "DOI" in text


def test_content_hash_changes_with_version_system_or_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_runtime_prompt_tree(tmp_path, version="1.0.0")
    monkeypatch.setattr(loader, "_resource_root", lambda: tmp_path)
    original_hash = load_prompt("query_understanding").content_hash

    _write_runtime_prompt_tree(tmp_path, version="1.0.1")
    version_hash = load_prompt("query_understanding").content_hash

    _write_runtime_prompt_tree(tmp_path, system_text="Changed system instructions")
    system_hash = load_prompt("query_understanding").content_hash

    _write_runtime_prompt_tree(
        tmp_path,
        user_text=f"Changed payload:\n{PAYLOAD_PLACEHOLDER}",
    )
    user_hash = load_prompt("query_understanding").content_hash

    assert len({original_hash, version_hash, system_hash, user_hash}) == 4


def _write_runtime_prompt_tree(
    root: Path,
    *,
    version: str = "1.0.0",
    system_text: str | None = "System instructions",
    user_text: str | None = None,
) -> None:
    prompt_dir = root / "query_understanding"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "query_understanding": {
            "version": version,
            "runtime_enabled": True,
            "system": "query_understanding/system.md",
            "user": "query_understanding/user.md",
        }
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    if system_text is not None:
        (prompt_dir / "system.md").write_text(system_text, encoding="utf-8")
    (prompt_dir / "user.md").write_text(
        user_text or f"Payload:\n{PAYLOAD_PLACEHOLDER}",
        encoding="utf-8",
    )
