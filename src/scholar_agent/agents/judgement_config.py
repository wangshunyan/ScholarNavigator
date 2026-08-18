"""确定性 Judgement 策略配置与稳定哈希。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scholar_agent.core.search_schemas import (
    JudgementPolicy,
    JudgementRuleConfig,
)


CURRENT_RULES_CONFIG = JudgementRuleConfig(
    config_version="current-rules-v1",
    lexical_normalization_policy="off",
    title_topic_weight=0.12,
    abstract_topic_weight=0.06,
    topic_max_score=0.45,
    title_must_have_weight=0.09,
    abstract_must_have_weight=0.045,
    must_have_max_score=0.24,
    title_method_weight=0.08,
    abstract_method_weight=0.04,
    method_max_score=0.18,
    title_dataset_weight=0.07,
    abstract_dataset_weight=0.035,
    dataset_max_score=0.12,
    title_domain_weight=0.05,
    abstract_domain_weight=0.025,
    domain_max_score=0.12,
    paper_type_match_weight=0.08,
    paper_type_max_score=0.16,
    paper_type_mismatch_penalty=0.08,
    venue_match_weight=0.10,
    venue_mismatch_penalty=0.03,
    temporal_match_weight=0.08,
    temporal_early_penalty=0.15,
    temporal_near_penalty=0.08,
    temporal_late_penalty=0.06,
    multi_dimension_bonus=0.02,
    multi_dimension_bonus_cap=0.08,
    insufficient_coverage_penalty=0.06,
    broad_topic_score_cap=0.68,
    explicit_dataset_penalty=0.08,
    missing_abstract_penalty=0.0,
    missing_metadata_penalty=0.0,
    highly_relevant_threshold=0.72,
    partially_relevant_threshold=0.45,
    weakly_relevant_threshold=0.25,
    minimum_evidence_count=0,
)

LEXICAL_NORMALIZATION_V1_CONFIG = CURRENT_RULES_CONFIG.model_copy(
    update={
        "config_version": "current-rules-lexical-normalization-v1",
        "lexical_normalization_policy": "lexical_normalization_v1",
    }
)

# 该常量只承载开发集冻结后产生的实验配置。产品默认仍由 policy 决定。
CALIBRATED_RULES_V1_CONFIG = CURRENT_RULES_CONFIG.model_copy(
    update={
        "config_version": "calibrated-rules-v1",
        "highly_relevant_threshold": 0.68,
        "title_topic_weight": 0.10,
    }
)


def judgement_config_hash(config: JudgementRuleConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_judgement_config(
    policy: JudgementPolicy,
    explicit: JudgementRuleConfig | None = None,
) -> JudgementRuleConfig:
    if explicit is not None:
        return explicit
    if policy == "calibrated_rules_v1":
        return CALIBRATED_RULES_V1_CONFIG
    return CURRENT_RULES_CONFIG


def load_judgement_config(path: Path | str) -> JudgementRuleConfig:
    return JudgementRuleConfig.model_validate_json(
        Path(path).expanduser().resolve().read_text(encoding="utf-8")
    )
