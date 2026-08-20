from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG, load_judgement_config


ROOT = Path(__file__).resolve().parents[1]
SOFT_CONFIG = ROOT / "benchmark" / "judgement_soft_current_rules_v1.json"


def test_soft_judgement_config_changes_only_the_reviewed_threshold() -> None:
    config = load_judgement_config(SOFT_CONFIG)
    assert config.partially_relevant_threshold == 0.35
    assert config.model_dump(exclude={"config_version", "partially_relevant_threshold"}) == (
        CURRENT_RULES_CONFIG.model_dump(
            exclude={"config_version", "partially_relevant_threshold"}
        )
    )


def test_soft_judgement_config_is_explicit_and_deterministic() -> None:
    raw = json.loads(SOFT_CONFIG.read_text(encoding="utf-8"))
    assert raw["config_version"] == "soft-current-rules-v1"
    assert raw["partially_relevant_threshold"] == 0.35


def test_linux_runner_wires_semantic_corpus_and_soft_config() -> None:
    script = (ROOT / "scripts" / "run_contest_benchmark.sh").read_text(encoding="utf-8")
    assert 'dense_reranker_soft' in script
    assert '"--judgement-config" "benchmark/judgement_soft_current_rules_v1.json"' in script
    assert '"datasets/semantic/pasa_papers_with_abstracts.jsonl"' in script
    assert '--reranker-device)' in script
    assert '"--local-hybrid-reranker-device" "$RERANKER_DEVICE"' in script


def test_windows_runner_wires_semantic_corpus_and_soft_config() -> None:
    script = (ROOT / "scripts" / "run_contest_benchmark.ps1").read_text(encoding="utf-8")
    assert '"dense_reranker_soft"' in script
    assert '"benchmark\\judgement_soft_current_rules_v1.json"' in script
    assert '"datasets\\semantic\\pasa_papers_with_abstracts.jsonl"' in script
    assert '"dense_reranker_soft", "dense_reranker_llm"' in script
    assert '[string]$RerankerDevice = "auto"' in script
    assert '"--local-hybrid-reranker-device", $RerankerDevice' in script
