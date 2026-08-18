"""Blinded, resumable LLM relevance judging for the frozen Record160 package.

This evaluator is deliberately separate from production retrieval and from the
human-label directory.  Only the already-public blind fields are sent to the
configured LLM.  Private arm mappings are opened only after a complete labels
lock exists, and are used solely by the offline scorer.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.evaluation.cluster_significance import (
    cluster_metric_statistics,
)
from scholar_agent.evaluation.execution_determinism import tree_signature
from scholar_agent.evaluation.full_swap_precision_annotation import (
    evaluate_full_swap_annotations,
)
from scholar_agent.evaluation.precision_annotation import (
    LABELS,
    PUBLIC_FIELDS,
    cohen_kappa,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.prompts.loader import validate_data_only_message_roles


CONTRACT_VERSION = "llm_relevance_judging_v1"
HARDENED_CONTRACT_VERSION = "llm_relevance_judging_v1_1"
SCHEMA_VERSION = "1"
GATE_NAME = "llm_relevance_judging_gate"
SCORE_SCOPE = "internal_llm_proxy_not_human_or_official"
EXIT_COMPLETED = 0
EXIT_INTEGRITY_VIOLATION = 2
EXIT_INCOMPLETE = 3
EXIT_USAGE_ERROR = 4
DEFAULT_SNAPSHOT_ROOT = (
    Path(__file__).resolve().parents[3] / "outputs" / "benchmark_snapshots"
)
DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "benchmark_runs"
    / "llm_relevance_judging_v1_record160"
)
DEFAULT_HARDENED_RUN_DIR = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "benchmark_runs"
    / "llm_relevance_judging_v1_1_record160"
)
LABEL_VALUES = tuple(str(value) for value in LABELS)
POSITIVE_LABELS = frozenset({"relevant", "partially_relevant"})
_LABEL_PATTERN = r"^item:[0-9a-f]{64}$"
_ROUND_IDS = ("independent_1", "independent_2")
_PLACEHOLDER = "{{payload}}"
_FORBIDDEN_RUNTIME_KEYS = frozenset(
    {"api_key", "authorization", "base_url", "headers", "token", "secret"}
)
_PROTOCOL_SPECS = {
    CONTRACT_VERSION: {
        "batch_size": 8,
        "max_concurrency": 4,
        "logical_retry_backoff_seconds": None,
        "complete_attempt_budget_in_one_run": False,
        "request_options": None,
        "structured_output": None,
    },
    HARDENED_CONTRACT_VERSION: {
        "batch_size": 1,
        "max_concurrency": 4,
        "logical_retry_backoff_seconds": 1.0,
        "complete_attempt_budget_in_one_run": True,
        "request_options": {"max_tokens": 1024, "timeout_seconds": 60.0},
        "structured_output": {
            "client_validation": "strict_pydantic_extra_forbid",
            "malformed_output_recovery": "forbidden",
            "provider_mode": "native_json_object",
            "top_level_adjudication_key": "decisions",
            "top_level_judge_key": "labels",
        },
    },
}


class LLMRelevanceJudgingError(RuntimeError):
    """A frozen input, run artifact, or CLI contract is invalid."""


class LLMRelevanceJudgingIncomplete(LLMRelevanceJudgingError):
    """The LLM is unavailable or one or more labels remain unresolved."""


class LLMJsonClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...


class BlindItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=_LABEL_PATTERN)
    query: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str
    year: int | None = None


class JudgeLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=_LABEL_PATTERN)
    label: Literal[
        "relevant",
        "partially_relevant",
        "not_relevant",
        "insufficient_information",
    ]
    evidence: str = Field(min_length=1, max_length=240)


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[JudgeLabel]


class AdjudicationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=_LABEL_PATTERN)
    final_label: Literal[
        "relevant",
        "partially_relevant",
        "not_relevant",
        "insufficient_information",
    ]
    evidence: str = Field(min_length=1, max_length=240)


class AdjudicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[AdjudicationDecision]


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _contract(protocol: Mapping[str, Any]) -> str:
    contract = str(protocol.get("contract") or "")
    if contract not in _PROTOCOL_SPECS:
        raise LLMRelevanceJudgingError("protocol_contract_invalid")
    return contract


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMRelevanceJudgingError("protocol_unreadable") from exc
    if not isinstance(value, dict):
        raise LLMRelevanceJudgingError("protocol_root_invalid")
    contract = str(value.get("contract") or "")
    if contract not in _PROTOCOL_SPECS:
        raise LLMRelevanceJudgingError("protocol_contract_invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise LLMRelevanceJudgingError("protocol_schema_invalid")
    if value.get("score_scope") != SCORE_SCOPE:
        raise LLMRelevanceJudgingError("protocol_score_scope_invalid")
    rubric = value.get("rubric") or {}
    if tuple(rubric.get("labels") or ()) != LABEL_VALUES:
        raise LLMRelevanceJudgingError("rubric_labels_drift")
    if set(rubric.get("positive_labels") or ()) != POSITIVE_LABELS:
        raise LLMRelevanceJudgingError("rubric_positive_labels_drift")
    judge = value.get("judge") or {}
    spec = _PROTOCOL_SPECS[contract]
    if judge.get("batch_size") != spec["batch_size"]:
        raise LLMRelevanceJudgingError("batch_size_drift")
    if judge.get("temperature") != 0:
        raise LLMRelevanceJudgingError("temperature_drift")
    if judge.get("max_logical_attempts_per_batch") != 2:
        raise LLMRelevanceJudgingError("attempt_limit_drift")
    if judge.get("max_concurrency") != spec["max_concurrency"]:
        raise LLMRelevanceJudgingError("concurrency_limit_drift")
    if (
        judge.get("logical_retry_backoff_seconds")
        != spec["logical_retry_backoff_seconds"]
    ):
        raise LLMRelevanceJudgingError("logical_retry_backoff_drift")
    if (
        judge.get("complete_attempt_budget_in_one_run", False)
        is not spec["complete_attempt_budget_in_one_run"]
    ):
        raise LLMRelevanceJudgingError("logical_attempt_execution_drift")
    if judge.get("structured_output") != spec["structured_output"]:
        raise LLMRelevanceJudgingError("structured_output_contract_drift")
    if judge.get("request_options") != spec["request_options"]:
        raise LLMRelevanceJudgingError("request_options_contract_drift")
    if judge.get("evidence_max_characters") != 240:
        raise LLMRelevanceJudgingError("evidence_limit_drift")
    if tuple(judge.get("independent_rounds") or ()) != _ROUND_IDS:
        raise LLMRelevanceJudgingError("independent_round_contract_drift")
    if contract == HARDENED_CONTRACT_VERSION:
        if judge["judge_prompt"].get("version") != "1.1.0" or judge[
            "adjudicator_prompt"
        ].get("version") != "1.1.0":
            raise LLMRelevanceJudgingError("hardened_prompt_version_drift")
        predecessor = value.get("predecessor") or {}
        if predecessor.get("inherit_locked_labels") is not False:
            raise LLMRelevanceJudgingError("predecessor_label_inheritance_forbidden")
        for name in ("calls", "manifest", "status"):
            predecessor_path = _repo_path(
                repository_root.resolve(),
                str(predecessor.get(f"{name}_path") or ""),
            )
            if sha256_file(predecessor_path) != predecessor.get(f"{name}_sha256"):
                raise LLMRelevanceJudgingError(
                    f"predecessor_hash_mismatch:{name}"
                )
    statistics_spec = value.get("statistics") or {}
    if statistics_spec != {
        "bootstrap_iterations": 20000,
        "cluster_assignment": (
            "reuse_frozen_component_id_without_recomputing_clusters"
        ),
        "cluster_bootstrap_seed": 20260722,
        "query_bootstrap_seed": 20260722,
        "top_k": 20,
    }:
        raise LLMRelevanceJudgingError("statistics_contract_drift")
    blinding = value.get("blinding") or {}
    if tuple(blinding.get("allowed_item_fields") or ()) != (
        "item_id",
        "query",
        "title",
        "abstract",
        "year",
    ):
        raise LLMRelevanceJudgingError("blind_field_contract_drift")
    root = repository_root.resolve()
    for section_name in ("current_package", "prior_package"):
        section = value["inputs"][section_name]
        for key, expected in section.items():
            if not key.endswith("_path"):
                continue
            hash_key = key.removesuffix("_path") + "_sha256"
            path_value = _repo_path(root, str(expected))
            if sha256_file(path_value) != section.get(hash_key):
                raise LLMRelevanceJudgingError(
                    f"input_hash_mismatch:{section_name}:{key}"
                )
    for section_name in ("record160_cases", "cluster_assignments"):
        section = value["inputs"][section_name]
        path_value = _repo_path(root, str(section["path"]))
        if sha256_file(path_value) != section["sha256"]:
            raise LLMRelevanceJudgingError(
                f"input_hash_mismatch:{section_name}"
            )
    for prompt_name in ("judge_prompt", "adjudicator_prompt"):
        prompt = judge[prompt_name]
        for role in ("system", "user"):
            prompt_path = _repo_path(root, str(prompt[f"{role}_path"]))
            if sha256_file(prompt_path) != prompt[f"{role}_sha256"]:
                raise LLMRelevanceJudgingError(
                    f"prompt_hash_mismatch:{prompt_name}:{role}"
                )
        user_text = _repo_path(root, str(prompt["user_path"])).read_text(
            encoding="utf-8"
        )
        if user_text.count(_PLACEHOLDER) != 1:
            raise LLMRelevanceJudgingError(
                f"prompt_placeholder_invalid:{prompt_name}"
            )
    return value


def prepare_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    _validate_run_directory(run_dir, repository_root)
    snapshot_before = tree_signature(DEFAULT_SNAPSHOT_ROOT)
    items, package_index = _build_blind_items(
        protocol, repository_root=repository_root
    )
    rows = [item.model_dump(mode="json") for item in items]
    forbidden = set(protocol["blinding"]["forbidden_fields"])
    _assert_judge_blinded_rows(
        rows,
        allowed_fields=protocol["blinding"]["allowed_item_fields"],
        forbidden_fields=forbidden,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "score_scope": SCORE_SCOPE,
        "state": "prepared",
        "item_count": len(rows),
        "item_set_sha256": stable_hash(sorted(row["item_id"] for row in rows)),
        "blind_view_sha256": stable_hash(rows),
        "package_index_sha256": stable_hash(package_index),
        "input_bindings_sha256": stable_hash(protocol["inputs"]),
        "prompt_bindings": {
            name: dict(protocol["judge"][name])
            for name in ("judge_prompt", "adjudicator_prompt")
        },
        "blinding": {
            "allowed_fields": list(protocol["blinding"]["allowed_item_fields"]),
            "forbidden_fields": list(protocol["blinding"]["forbidden_fields"]),
            "recursive_forbidden_field_match_count": 0,
            "private_mapping_written": False,
        },
        "execution": _execution_counts(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_once_jsonl(run_dir / "blind_view.jsonl", rows)
    _write_once_json(run_dir / "prepared_manifest.json", manifest)
    if snapshot_before != tree_signature(DEFAULT_SNAPSHOT_ROOT):
        raise LLMRelevanceJudgingError("snapshot_tree_modified")
    return _status_report(
        "completed",
        EXIT_COMPLETED,
        stage="prepare",
        details={
            "item_count": len(rows),
            "blind_view_sha256": manifest["blind_view_sha256"],
            "current_package_item_count": sum(
                item["scope"] == "current" for item in package_index
            ),
            "prior_package_item_count": sum(
                item["scope"] == "prior" for item in package_index
            ),
        },
        contract=_contract(protocol),
    )


def run_judge_round(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    round_id: str,
    client: LLMJsonClient,
    runtime_binding: Mapping[str, Any],
    max_batches: int | None = None,
    client_factory: Callable[[], LLMJsonClient] | None = None,
) -> dict[str, Any]:
    _validate_run_directory(run_dir, repository_root)
    if round_id not in _ROUND_IDS:
        raise LLMRelevanceJudgingError("round_id_invalid")
    items, prepared = _load_prepared(
        protocol, repository_root=repository_root, run_dir=run_dir
    )
    binding = _bind_runtime(
        run_dir,
        runtime_binding,
        contract=_contract(protocol),
    )
    batches = _chunks(items, int(protocol["judge"]["batch_size"]))
    directory = run_dir / "rounds" / round_id
    directory.mkdir(parents=True, exist_ok=True)
    pending: list[
        tuple[int, list[BlindItem], Path, dict[str, Any] | None, list[dict[str, str]]]
    ] = []
    for index, batch in enumerate(batches):
        batch_path = directory / f"batch_{index:05d}.json"
        existing = _load_existing_batch(
            batch_path,
            mode="judge",
            phase=round_id,
            index=index,
            items=batch,
            runtime_binding_sha256=stable_hash(binding),
            contract=_contract(protocol),
        )
        if existing is not None and existing["status"] == "locked_success":
            continue
        if existing is not None and len(existing["attempts"]) >= int(
            protocol["judge"]["max_logical_attempts_per_batch"]
        ):
            continue
        if max_batches is not None and len(pending) >= max_batches:
            continue
        messages = _judge_messages(
            protocol,
            repository_root=repository_root,
            items=batch,
        )
        pending.append((index, batch, batch_path, existing, messages))
    called = _execute_batch_work(
        pending,
        client=client,
        client_factory=client_factory,
        mode="judge",
        phase=round_id,
        runtime_binding_sha256=stable_hash(binding),
        protocol=protocol,
    )
    coverage = _round_coverage(
        directory,
        mode="judge",
        phase=round_id,
        batches=batches,
        runtime_binding_sha256=stable_hash(binding),
        protocol=protocol,
    )
    status, exit_code = _coverage_status(coverage)
    return _status_report(
        status,
        exit_code,
        stage="judge",
        details={
            "round": round_id,
            "model": binding["model"],
            "provider": binding["provider"],
            "prompt_version": protocol["judge"]["judge_prompt"]["version"],
            "prepared_view_sha256": prepared["blind_view_sha256"],
            "batch_count": len(batches),
            "called_batch_count": called,
            **coverage,
        },
        contract=_contract(protocol),
    )


def run_adjudication(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    client: LLMJsonClient,
    runtime_binding: Mapping[str, Any],
    max_batches: int | None = None,
    client_factory: Callable[[], LLMJsonClient] | None = None,
) -> dict[str, Any]:
    _validate_run_directory(run_dir, repository_root)
    items, prepared = _load_prepared(
        protocol, repository_root=repository_root, run_dir=run_dir
    )
    binding = _bind_runtime(
        run_dir,
        runtime_binding,
        contract=_contract(protocol),
    )
    first = _locked_round_labels(protocol, run_dir, "independent_1", items)
    second = _locked_round_labels(protocol, run_dir, "independent_2", items)
    disagreements = [
        item
        for item in items
        if first[item.item_id]["label"] != second[item.item_id]["label"]
    ]
    batches = _chunks(disagreements, int(protocol["judge"]["batch_size"]))
    directory = run_dir / "adjudication"
    directory.mkdir(parents=True, exist_ok=True)
    pending: list[
        tuple[int, list[BlindItem], Path, dict[str, Any] | None, list[dict[str, str]]]
    ] = []
    for index, batch in enumerate(batches):
        batch_path = directory / f"batch_{index:05d}.json"
        existing = _load_existing_batch(
            batch_path,
            mode="adjudicate",
            phase="adjudication",
            index=index,
            items=batch,
            runtime_binding_sha256=stable_hash(binding),
            contract=_contract(protocol),
        )
        if existing is not None and existing["status"] == "locked_success":
            continue
        if existing is not None and len(existing["attempts"]) >= int(
            protocol["judge"]["max_logical_attempts_per_batch"]
        ):
            continue
        if max_batches is not None and len(pending) >= max_batches:
            continue
        messages = _adjudication_messages(
            protocol,
            repository_root=repository_root,
            items=batch,
            first=first,
            second=second,
        )
        pending.append((index, batch, batch_path, existing, messages))
    called = _execute_batch_work(
        pending,
        client=client,
        client_factory=client_factory,
        mode="adjudicate",
        phase="adjudication",
        runtime_binding_sha256=stable_hash(binding),
        protocol=protocol,
    )
    coverage = _round_coverage(
        directory,
        mode="adjudicate",
        phase="adjudication",
        batches=batches,
        runtime_binding_sha256=stable_hash(binding),
        protocol=protocol,
    )
    status, exit_code = _coverage_status(coverage)
    if exit_code == EXIT_COMPLETED:
        final_records = _resolved_records(protocol, run_dir, items)
        lock = _labels_lock(
            protocol,
            run_dir=run_dir,
            prepared=prepared,
            runtime_binding=binding,
            records=final_records,
        )
        _write_once_json(run_dir / "labels_lock.json", lock)
    return _status_report(
        status,
        exit_code,
        stage="adjudicate",
        details={
            "model": binding["model"],
            "provider": binding["provider"],
            "prompt_version": protocol["judge"]["adjudicator_prompt"]["version"],
            "item_count": len(items),
            "disagreement_count": len(disagreements),
            "batch_count": len(batches),
            "called_batch_count": called,
            **coverage,
        },
        contract=_contract(protocol),
    )


def verify_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    _validate_run_directory(run_dir, repository_root)
    items, prepared = _load_prepared(
        protocol, repository_root=repository_root, run_dir=run_dir
    )
    binding_path = run_dir / "runtime_binding.json"
    if not binding_path.is_file():
        return _status_report(
            "incomplete_or_llm_unavailable",
            EXIT_INCOMPLETE,
            stage="verify",
            details={
                "item_count": len(items),
                "blind_view_sha256": prepared["blind_view_sha256"],
                "reason": "runtime_binding_missing",
            },
            contract=_contract(protocol),
        )
    binding = _read_json(binding_path)
    _validate_runtime_binding(binding, contract=_contract(protocol))
    details: dict[str, Any] = {
        "item_count": len(items),
        "blind_view_sha256": prepared["blind_view_sha256"],
        "rounds": {},
    }
    complete = True
    terminal_schema_failures = 0
    for round_id in _ROUND_IDS:
        batches = _chunks(items, int(protocol["judge"]["batch_size"]))
        coverage = _round_coverage(
            run_dir / "rounds" / round_id,
            mode="judge",
            phase=round_id,
            batches=batches,
            runtime_binding_sha256=stable_hash(binding),
            protocol=protocol,
        )
        details["rounds"][round_id] = coverage
        complete &= coverage["locked_item_count"] == len(items)
        terminal_schema_failures += coverage["terminal_schema_failure_count"]
    if not complete:
        return _status_report(
            "integrity_or_schema_violation"
            if terminal_schema_failures
            else "incomplete_or_llm_unavailable",
            EXIT_INTEGRITY_VIOLATION if terminal_schema_failures else EXIT_INCOMPLETE,
            stage="verify",
            details=details,
            contract=_contract(protocol),
        )
    first = _locked_round_labels(protocol, run_dir, "independent_1", items)
    second = _locked_round_labels(protocol, run_dir, "independent_2", items)
    disagreements = [
        item
        for item in items
        if first[item.item_id]["label"] != second[item.item_id]["label"]
    ]
    adjudication_batches = _chunks(
        disagreements, int(protocol["judge"]["batch_size"])
    )
    adjudication_coverage = _round_coverage(
        run_dir / "adjudication",
        mode="adjudicate",
        phase="adjudication",
        batches=adjudication_batches,
        runtime_binding_sha256=stable_hash(binding),
        protocol=protocol,
    )
    details["adjudication"] = adjudication_coverage
    details["disagreement_count"] = len(disagreements)
    if adjudication_coverage["locked_item_count"] != len(disagreements):
        exit_code = (
            EXIT_INTEGRITY_VIOLATION
            if adjudication_coverage["terminal_schema_failure_count"]
            else EXIT_INCOMPLETE
        )
        return _status_report(
            "integrity_or_schema_violation"
            if exit_code == EXIT_INTEGRITY_VIOLATION
            else "incomplete_or_llm_unavailable",
            exit_code,
            stage="verify",
            details=details,
            contract=_contract(protocol),
        )
    records = _resolved_records(protocol, run_dir, items)
    observed = _labels_lock(
        protocol,
        run_dir=run_dir,
        prepared=prepared,
        runtime_binding=binding,
        records=records,
    )
    lock_path = run_dir / "labels_lock.json"
    if not lock_path.is_file():
        return _status_report(
            "incomplete_or_llm_unavailable",
            EXIT_INCOMPLETE,
            stage="verify",
            details={**details, "reason": "labels_lock_missing"},
            contract=_contract(protocol),
        )
    if _read_json(lock_path) != observed:
        raise LLMRelevanceJudgingError("labels_lock_mismatch")
    return _status_report(
        "completed",
        EXIT_COMPLETED,
        stage="verify",
        details={
            **details,
            "labels_sha256": observed["labels_sha256"],
            "locked_batch_tree_sha256": observed["locked_batch_tree_sha256"],
        },
        contract=_contract(protocol),
    )


def score_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    publish_dir: Path | None = None,
) -> dict[str, Any]:
    _validate_run_directory(run_dir, repository_root)
    verification = verify_run(
        protocol, repository_root=repository_root, run_dir=run_dir
    )
    if verification["exit_code"] != EXIT_COMPLETED:
        raise LLMRelevanceJudgingIncomplete("labels_not_locked")
    items, prepared = _load_prepared(
        protocol, repository_root=repository_root, run_dir=run_dir
    )
    records = _resolved_records(protocol, run_dir, items)
    score = _score_locked_records(
        protocol,
        repository_root=repository_root,
        records=records,
    )
    usage = _usage_summary(run_dir)
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "score_scope": SCORE_SCOPE,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "model": _read_json(run_dir / "runtime_binding.json")["model"],
        "provider": _read_json(run_dir / "runtime_binding.json")["provider"],
        "prompt_versions": {
            "judge": protocol["judge"]["judge_prompt"]["version"],
            "adjudicator": protocol["judge"]["adjudicator_prompt"]["version"],
        },
        "package": {
            "item_count": len(items),
            "blind_view_sha256": prepared["blind_view_sha256"],
            "labels_sha256": stable_hash(records),
        },
        "review": score,
        "usage": usage,
        "execution": {
            **_execution_counts(),
            "llm_logical_call_count": usage["logical_call_count"],
            "llm_http_attempt_count": usage["known_http_attempt_count"],
        },
        "warnings": [
            "Internal LLM proxy only; not human Precision or an official score.",
            "Absolute arm Precision@20 is unavailable because shared Top-20 items were not in the frozen change-only package.",
        ],
    }
    result["result_payload_sha256"] = stable_hash(result)
    _atomic_write_json(run_dir / "score.json", result)
    if publish_dir is not None:
        _publish_run(
            protocol,
            repository_root=repository_root,
            run_dir=run_dir,
            publish_dir=publish_dir,
            records=records,
            result=result,
        )
    return result


def publish_incomplete_audit(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    publish_dir: Path,
) -> dict[str, Any]:
    """Publish failure/coverage evidence without exposing partial labels.

    A terminally incomplete run is still useful operational evidence, but it
    must never unlock the private arm mapping or produce proxy statistics.
    """

    verification = verify_run(
        protocol,
        repository_root=repository_root,
        run_dir=run_dir,
    )
    if verification["exit_code"] == EXIT_COMPLETED:
        raise LLMRelevanceJudgingError("completed_run_requires_score_publish")
    _validate_publish_directory(publish_dir, repository_root)
    binding_path = run_dir / "runtime_binding.json"
    binding = _read_json(binding_path) if binding_path.is_file() else None
    calls = _call_audit_rows(run_dir)
    usage = _usage_summary(run_dir)
    status = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "score_scope": SCORE_SCOPE,
        "status": verification["status"],
        "exit_code": verification["exit_code"],
        "verification": verification,
        "runtime_binding": binding,
        "prompt_versions": {
            "judge": protocol["judge"]["judge_prompt"]["version"],
            "adjudicator": protocol["judge"]["adjudicator_prompt"]["version"],
        },
        "usage": usage,
        "labels_locked": False,
        "private_mapping_opened": False,
        "statistics": None,
        "reason": "strict_schema_coverage_incomplete",
        "execution": {
            **_execution_counts(),
            "llm_logical_call_count": usage["logical_call_count"],
            "llm_http_attempt_count": usage["known_http_attempt_count"],
        },
        "warnings": [
            "No LLM-proxy Precision or paired effect was computed.",
            "This is neither human Precision nor an official score.",
        ],
    }
    status["status_payload_sha256"] = stable_hash(status)
    publish_dir.mkdir(parents=True, exist_ok=True)
    calls_path = publish_dir / "calls.jsonl"
    status_path = publish_dir / "status.json"
    _write_once_jsonl(calls_path, calls)
    _write_once_json(status_path, status)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "score_scope": SCORE_SCOPE,
        "protocol_sha256": stable_hash(protocol),
        "state": "incomplete_no_unblinding",
        "calls": {
            "path": "calls.jsonl",
            "record_count": len(calls),
            "size_bytes": calls_path.stat().st_size,
            "sha256": sha256_file(calls_path),
            "contains_prompt_or_response_content": False,
        },
        "status": {
            "path": "status.json",
            "size_bytes": status_path.stat().st_size,
            "sha256": sha256_file(status_path),
        },
        "labels_file": None,
        "statistics_file": None,
        "human_label_directory_modified": False,
        "default_strategy_changed": False,
    }
    _write_once_json(publish_dir / "manifest.json", manifest)
    return status


def runtime_binding_for_client(
    client: Any,
    *,
    provider: str,
    model: str,
    request_options: Mapping[str, int | float],
    contract: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    if contract not in _PROTOCOL_SPECS:
        raise LLMRelevanceJudgingError("runtime_binding_contract_invalid")
    endpoint = str(getattr(client, "base_url", ""))
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "provider": provider,
        "model": model,
        "provider_endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "request_options": {
            "max_tokens": int(request_options["max_tokens"]),
            "timeout_seconds": float(request_options["timeout_seconds"]),
        },
    }


def _build_blind_items(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> tuple[list[BlindItem], list[dict[str, str]]]:
    inputs = protocol["inputs"]
    current = inputs["current_package"]
    prior = inputs["prior_package"]
    current_rows = _read_jsonl(_repo_path(repository_root, current["public_path"]))
    mapping = _read_json(_repo_path(repository_root, current["mapping_path"]))
    prior_rows = _read_jsonl(_repo_path(repository_root, prior["public_path"]))
    if len(current_rows) != int(current["expected_item_count"]):
        raise LLMRelevanceJudgingError("current_package_item_count_drift")
    current_by_id = _public_index(current_rows)
    prior_by_id = _public_index(prior_rows)
    referenced_prior = sorted(
        {
            str(item["prior_sample_id"])
            for item in mapping.get("prior_package_overlaps") or []
        }
    )
    if len(referenced_prior) != int(prior["expected_referenced_item_count"]):
        raise LLMRelevanceJudgingError("prior_reference_count_drift")
    missing = set(referenced_prior) - set(prior_by_id)
    if missing:
        raise LLMRelevanceJudgingIncomplete("prior_visible_metadata_missing")
    records: list[tuple[str, str, Mapping[str, Any]]] = [
        ("current", sample_id, row)
        for sample_id, row in current_by_id.items()
    ] + [
        ("prior", sample_id, prior_by_id[sample_id])
        for sample_id in referenced_prior
    ]
    items: list[BlindItem] = []
    package_index: list[dict[str, str]] = []
    for scope, sample_id, row in records:
        item_id = _opaque_item_id(
            scope,
            sample_id,
            contract=_contract(protocol),
            package_sha256=(
                current["public_sha256"]
                if scope == "current"
                else prior["public_sha256"]
            ),
        )
        items.append(
            BlindItem(
                item_id=item_id,
                query=str(row["query"]),
                title=str(row["title"]),
                abstract=str(row.get("abstract") or ""),
                year=row.get("year"),
            )
        )
        package_index.append(
            {"item_id": item_id, "scope": scope, "sample_id": sample_id}
        )
    items.sort(key=lambda item: item.item_id)
    package_index.sort(key=lambda item: item["item_id"])
    if len({item.item_id for item in items}) != len(items):
        raise LLMRelevanceJudgingError("opaque_item_identity_collision")
    return items, package_index


def _public_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    allowed = set(PUBLIC_FIELDS)
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if set(row) != allowed:
            raise LLMRelevanceJudgingError("public_blind_row_field_mismatch")
        try:
            sample_id = str(row["sample_id"])
            BlindItem(
                item_id="item:" + "0" * 64,
                query=str(row["query"]),
                title=str(row["title"]),
                abstract=str(row.get("abstract") or ""),
                year=row.get("year"),
            )
        except (KeyError, ValidationError) as exc:
            raise LLMRelevanceJudgingError("public_blind_row_invalid") from exc
        if sample_id in output:
            raise LLMRelevanceJudgingError("duplicate_public_sample_id")
        output[sample_id] = row
    return output


def _opaque_item_id(
    scope: str,
    sample_id: str,
    *,
    contract: str,
    package_sha256: str,
) -> str:
    payload = "\x1f".join((contract, scope, package_sha256, sample_id))
    return "item:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_prepared(
    protocol: Mapping[str, Any], *, repository_root: Path, run_dir: Path
) -> tuple[list[BlindItem], dict[str, Any]]:
    expected, package_index = _build_blind_items(
        protocol, repository_root=repository_root
    )
    rows = _read_jsonl(run_dir / "blind_view.jsonl")
    try:
        observed = [BlindItem.model_validate(row) for row in rows]
    except ValidationError as exc:
        raise LLMRelevanceJudgingError("prepared_view_schema_invalid") from exc
    if [item.model_dump(mode="json") for item in observed] != [
        item.model_dump(mode="json") for item in expected
    ]:
        raise LLMRelevanceJudgingError("prepared_view_mismatch")
    manifest = _read_json(run_dir / "prepared_manifest.json")
    if manifest.get("blind_view_sha256") != stable_hash(rows):
        raise LLMRelevanceJudgingError("prepared_view_hash_mismatch")
    if manifest.get("package_index_sha256") != stable_hash(package_index):
        raise LLMRelevanceJudgingError("prepared_package_index_mismatch")
    return observed, manifest


def _bind_runtime(
    run_dir: Path,
    binding: Mapping[str, Any],
    *,
    contract: str,
) -> dict[str, Any]:
    value = dict(binding)
    _validate_runtime_binding(value, contract=contract)
    path = run_dir / "runtime_binding.json"
    if path.exists():
        if _read_json(path) != value:
            raise LLMRelevanceJudgingError("runtime_binding_drift")
    else:
        _atomic_write_json(path, value)
    return value


def _validate_runtime_binding(value: Mapping[str, Any], *, contract: str) -> None:
    if value.get("contract") != contract or value.get("schema_version") != SCHEMA_VERSION:
        raise LLMRelevanceJudgingError("runtime_binding_contract_invalid")
    if not str(value.get("provider") or "") or not str(value.get("model") or ""):
        raise LLMRelevanceJudgingError("runtime_binding_provider_invalid")
    if not _is_sha256(value.get("provider_endpoint_sha256")):
        raise LLMRelevanceJudgingError("runtime_binding_endpoint_invalid")
    if set(value) & _FORBIDDEN_RUNTIME_KEYS:
        raise LLMRelevanceJudgingError("runtime_binding_contains_sensitive_field")
    options = value.get("request_options")
    if not isinstance(options, Mapping):
        raise LLMRelevanceJudgingError("runtime_binding_options_invalid")
    if float(options.get("timeout_seconds") or 0) <= 0:
        raise LLMRelevanceJudgingError("runtime_binding_timeout_invalid")
    if int(options.get("max_tokens") or 0) <= 0:
        raise LLMRelevanceJudgingError("runtime_binding_max_tokens_invalid")
    expected_options = _PROTOCOL_SPECS[contract]["request_options"]
    if expected_options is not None and dict(options) != expected_options:
        raise LLMRelevanceJudgingError("runtime_binding_request_options_drift")


def _judge_messages(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    items: Sequence[BlindItem],
) -> list[dict[str, str]]:
    rubric_path = _repo_path(
        repository_root,
        protocol["inputs"]["current_package"]["rubric_path"],
    )
    rubric = _read_json(rubric_path)
    payload = {
        "rubric": {
            "labels": list(protocol["rubric"]["labels"]),
            "definitions": rubric["definitions"],
        },
        "items": [item.model_dump(mode="json") for item in items],
    }
    return _render_messages(
        protocol["judge"]["judge_prompt"],
        repository_root=repository_root,
        payload=payload,
    )


def _adjudication_messages(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    items: Sequence[BlindItem],
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    rubric = _read_json(
        _repo_path(
            repository_root,
            protocol["inputs"]["current_package"]["rubric_path"],
        )
    )
    payload = {
        "rubric": {
            "labels": list(protocol["rubric"]["labels"]),
            "definitions": rubric["definitions"],
        },
        "items": [
            {
                **item.model_dump(mode="json"),
                "anonymous_reviews": [
                    {
                        "label": first[item.item_id]["label"],
                        "evidence": first[item.item_id]["evidence"],
                    },
                    {
                        "label": second[item.item_id]["label"],
                        "evidence": second[item.item_id]["evidence"],
                    },
                ],
            }
            for item in items
        ],
    }
    return _render_messages(
        protocol["judge"]["adjudicator_prompt"],
        repository_root=repository_root,
        payload=payload,
    )


def _render_messages(
    prompt: Mapping[str, Any],
    *,
    repository_root: Path,
    payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = _repo_path(repository_root, prompt["system_path"]).read_text(
        encoding="utf-8"
    ).strip()
    user = _repo_path(repository_root, prompt["user_path"]).read_text(
        encoding="utf-8"
    ).strip()
    envelope = {
        "boundary": {
            "contract": "untrusted_metadata_isolation_v1",
            "metadata_role": "untrusted_data",
            "instruction_capability": False,
        },
        "payload": payload,
    }
    serialized = json.dumps(
        envelope,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for character, escaped in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
    ):
        serialized = serialized.replace(character, escaped)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user.replace(_PLACEHOLDER, serialized)},
    ]
    validate_data_only_message_roles(messages)
    return messages


def _invoke_batch(
    client: LLMJsonClient,
    messages: list[dict[str, str]],
    *,
    expected_ids: Sequence[str],
    mode: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    input_sha256 = stable_hash(messages)
    try:
        raw = client.chat_json(
            messages,
            temperature=float(protocol["judge"]["temperature"]),
            timeout=None,
        )
    except Exception as exc:  # noqa: BLE001 - converted to a non-sensitive code
        return {
            "status": "provider_failure",
            "input_sha256": input_sha256,
            "response_sha256": None,
            "response": None,
            "failure": _safe_provider_failure(exc),
            "usage": _last_call_usage(client),
            "diagnostics": _last_call_diagnostics(client),
        }
    response_sha256 = stable_hash(raw)
    try:
        normalized = _validate_response(raw, expected_ids=expected_ids, mode=mode)
    except LLMRelevanceJudgingError as exc:
        return {
            "status": "schema_failure",
            "input_sha256": input_sha256,
            "response_sha256": response_sha256,
            "response": None,
            "failure": {"code": str(exc), "kind": "schema_failure"},
            "usage": _last_call_usage(client),
            "diagnostics": _last_call_diagnostics(client),
        }
    return {
        "status": "success",
        "input_sha256": input_sha256,
        "response_sha256": stable_hash(normalized),
        "response": normalized,
        "failure": None,
        "usage": _last_call_usage(client),
        "diagnostics": _last_call_diagnostics(client),
    }


def _execute_batch_work(
    work: Sequence[
        tuple[
            int,
            list[BlindItem],
            Path,
            dict[str, Any] | None,
            list[dict[str, str]],
        ]
    ],
    *,
    client: LLMJsonClient,
    client_factory: Callable[[], LLMJsonClient] | None,
    mode: str,
    phase: str,
    runtime_binding_sha256: str,
    protocol: Mapping[str, Any],
) -> int:
    """Execute stable microbatches with isolated per-call clients.

    Concurrent completion order cannot affect the batch identity or file path.
    A caller that does not provide a factory retains the original serial path,
    which is useful for small offline fakes and compatibility callers.
    """

    if not work:
        return 0

    def invoke(
        entry: tuple[
            int,
            list[BlindItem],
            Path,
            dict[str, Any] | None,
            list[dict[str, str]],
        ]
    ) -> tuple[int, Path, dict[str, Any]]:
        index, items, path, existing, messages = entry
        current_client = client_factory() if client_factory is not None else client
        payload = existing
        maximum = int(protocol["judge"]["max_logical_attempts_per_batch"])
        run_full_budget = bool(
            protocol["judge"].get("complete_attempt_budget_in_one_run", False)
        )
        while payload is None or (
            payload["status"] != "locked_success"
            and len(payload["attempts"]) < maximum
        ):
            if payload is not None:
                backoff = float(
                    protocol["judge"].get("logical_retry_backoff_seconds") or 0
                )
                if backoff > 0:
                    time.sleep(backoff)
            attempt = _invoke_batch(
                current_client,
                messages,
                expected_ids=[item.item_id for item in items],
                mode=mode,
                protocol=protocol,
            )
            payload = _update_batch(
                payload,
                mode=mode,
                phase=phase,
                index=index,
                items=items,
                runtime_binding_sha256=runtime_binding_sha256,
                attempt=attempt,
                contract=_contract(protocol),
            )
            _atomic_write_json(path, payload)
            if payload["status"] == "locked_success" or not run_full_budget:
                break
        return index, path, payload

    concurrency = (
        int(protocol["judge"]["max_concurrency"])
        if client_factory is not None
        else 1
    )
    if concurrency == 1:
        for entry in work:
            _, path, payload = invoke(entry)
            _atomic_write_json(path, payload)
        return len(work)

    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="llm-relevance-judge",
    ) as executor:
        futures = [executor.submit(invoke, entry) for entry in work]
        for future in as_completed(futures):
            _, path, payload = future.result()
            _atomic_write_json(path, payload)
    return len(work)


def _validate_response(
    raw: Mapping[str, Any], *, expected_ids: Sequence[str], mode: str
) -> dict[str, Any]:
    try:
        if mode == "judge":
            parsed = JudgeResponse.model_validate(raw)
            rows = [item.model_dump(mode="json") for item in parsed.labels]
            key = "labels"
        else:
            parsed = AdjudicationResponse.model_validate(raw)
            rows = [item.model_dump(mode="json") for item in parsed.decisions]
            key = "decisions"
    except ValidationError as exc:
        raise LLMRelevanceJudgingError("llm_response_schema_invalid") from exc
    observed = [str(item["item_id"]) for item in rows]
    if len(observed) != len(set(observed)):
        raise LLMRelevanceJudgingError("llm_response_duplicate_item")
    if observed != list(expected_ids):
        raise LLMRelevanceJudgingError("llm_response_batch_mismatch")
    return {key: rows}


def _update_batch(
    existing: Mapping[str, Any] | None,
    *,
    mode: str,
    phase: str,
    index: int,
    items: Sequence[BlindItem],
    runtime_binding_sha256: str,
    attempt: Mapping[str, Any],
    contract: str,
) -> dict[str, Any]:
    attempts = list(existing.get("attempts") or []) if existing else []
    attempt_row = {"attempt": len(attempts) + 1, **dict(attempt)}
    attempts.append(attempt_row)
    status = "locked_success" if attempt["status"] == "success" else "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "mode": mode,
        "phase": phase,
        "batch_index": index,
        "batch_identity": _batch_identity(
            mode,
            phase,
            index,
            items,
            contract=contract,
        ),
        "item_ids": [item.item_id for item in items],
        "item_set_sha256": stable_hash([item.item_id for item in items]),
        "runtime_binding_sha256": runtime_binding_sha256,
        "status": status,
        "attempts": attempts,
        "locked_response": attempt["response"] if status == "locked_success" else None,
        "locked_response_sha256": (
            attempt["response_sha256"] if status == "locked_success" else None
        ),
    }


def _load_existing_batch(
    path: Path,
    *,
    mode: str,
    phase: str,
    index: int,
    items: Sequence[BlindItem],
    runtime_binding_sha256: str,
    contract: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read_json(path)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "mode": mode,
        "phase": phase,
        "batch_index": index,
        "batch_identity": _batch_identity(
            mode,
            phase,
            index,
            items,
            contract=contract,
        ),
        "item_ids": [item.item_id for item in items],
        "item_set_sha256": stable_hash([item.item_id for item in items]),
        "runtime_binding_sha256": runtime_binding_sha256,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise LLMRelevanceJudgingError("batch_binding_mismatch")
    if value.get("status") not in {"locked_success", "failed"}:
        raise LLMRelevanceJudgingError("batch_status_invalid")
    attempts = value.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise LLMRelevanceJudgingError("batch_attempts_invalid")
    if [item.get("attempt") for item in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise LLMRelevanceJudgingError("batch_attempt_sequence_invalid")
    if value["status"] == "locked_success":
        response = value.get("locked_response")
        if stable_hash(response) != value.get("locked_response_sha256"):
            raise LLMRelevanceJudgingError("locked_response_hash_mismatch")
        normalized = _validate_response(
            response,
            expected_ids=[item.item_id for item in items],
            mode=mode,
        )
        if normalized != response:
            raise LLMRelevanceJudgingError("locked_response_not_canonical")
        if attempts[-1].get("status") != "success":
            raise LLMRelevanceJudgingError("locked_response_attempt_mismatch")
    elif value.get("locked_response") is not None:
        raise LLMRelevanceJudgingError("failed_batch_has_locked_response")
    return value


def _round_coverage(
    directory: Path,
    *,
    mode: str,
    phase: str,
    batches: Sequence[Sequence[BlindItem]],
    runtime_binding_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    locked_items = 0
    missing_batches = 0
    retryable_failures = 0
    terminal_provider_failures = 0
    terminal_schema_failures = 0
    missing_item_ids: list[str] = []
    retryable_item_ids: list[str] = []
    terminal_provider_item_ids: list[str] = []
    terminal_schema_item_ids: list[str] = []
    expected_paths = {f"batch_{index:05d}.json" for index in range(len(batches))}
    observed_paths = (
        {path.name for path in directory.glob("batch_*.json")}
        if directory.is_dir()
        else set()
    )
    if observed_paths - expected_paths:
        raise LLMRelevanceJudgingError("unexpected_batch_file")
    maximum = int(protocol["judge"]["max_logical_attempts_per_batch"])
    for index, items in enumerate(batches):
        value = _load_existing_batch(
            directory / f"batch_{index:05d}.json",
            mode=mode,
            phase=phase,
            index=index,
            items=items,
            runtime_binding_sha256=runtime_binding_sha256,
            contract=_contract(protocol),
        )
        if value is None:
            missing_batches += 1
            missing_item_ids.extend(item.item_id for item in items)
            continue
        if value["status"] == "locked_success":
            locked_items += len(items)
            continue
        attempts = value["attempts"]
        if len(attempts) < maximum:
            retryable_failures += 1
            retryable_item_ids.extend(item.item_id for item in items)
        elif attempts[-1]["status"] == "schema_failure":
            terminal_schema_failures += 1
            terminal_schema_item_ids.extend(item.item_id for item in items)
        else:
            terminal_provider_failures += 1
            terminal_provider_item_ids.extend(item.item_id for item in items)
    return {
        "expected_item_count": sum(len(items) for items in batches),
        "locked_item_count": locked_items,
        "missing_batch_count": missing_batches,
        "retryable_failure_count": retryable_failures,
        "terminal_provider_failure_count": terminal_provider_failures,
        "terminal_schema_failure_count": terminal_schema_failures,
        "missing_item_ids": missing_item_ids,
        "retryable_failure_item_ids": retryable_item_ids,
        "terminal_provider_failure_item_ids": terminal_provider_item_ids,
        "terminal_schema_failure_item_ids": terminal_schema_item_ids,
    }


def _coverage_status(coverage: Mapping[str, Any]) -> tuple[str, int]:
    if coverage["terminal_schema_failure_count"]:
        return "integrity_or_schema_violation", EXIT_INTEGRITY_VIOLATION
    if coverage["locked_item_count"] == coverage["expected_item_count"]:
        return "completed", EXIT_COMPLETED
    return "incomplete_or_llm_unavailable", EXIT_INCOMPLETE


def _locked_round_labels(
    protocol: Mapping[str, Any],
    run_dir: Path,
    round_id: str,
    items: Sequence[BlindItem],
) -> dict[str, dict[str, Any]]:
    binding = _read_json(run_dir / "runtime_binding.json")
    batches = _chunks(items, int(protocol["judge"]["batch_size"]))
    output: dict[str, dict[str, Any]] = {}
    for index, batch in enumerate(batches):
        value = _load_existing_batch(
            run_dir / "rounds" / round_id / f"batch_{index:05d}.json",
            mode="judge",
            phase=round_id,
            index=index,
            items=batch,
            runtime_binding_sha256=stable_hash(binding),
            contract=_contract(protocol),
        )
        if value is None or value["status"] != "locked_success":
            raise LLMRelevanceJudgingIncomplete(f"round_incomplete:{round_id}")
        for row in value["locked_response"]["labels"]:
            output[str(row["item_id"])] = dict(row)
    if set(output) != {item.item_id for item in items}:
        raise LLMRelevanceJudgingError("round_label_coverage_mismatch")
    return output


def _resolved_records(
    protocol: Mapping[str, Any], run_dir: Path, items: Sequence[BlindItem]
) -> list[dict[str, Any]]:
    first = _locked_round_labels(protocol, run_dir, "independent_1", items)
    second = _locked_round_labels(protocol, run_dir, "independent_2", items)
    disagreements = [
        item
        for item in items
        if first[item.item_id]["label"] != second[item.item_id]["label"]
    ]
    decisions: dict[str, dict[str, Any]] = {}
    binding = _read_json(run_dir / "runtime_binding.json")
    for index, batch in enumerate(
        _chunks(disagreements, int(protocol["judge"]["batch_size"]))
    ):
        value = _load_existing_batch(
            run_dir / "adjudication" / f"batch_{index:05d}.json",
            mode="adjudicate",
            phase="adjudication",
            index=index,
            items=batch,
            runtime_binding_sha256=stable_hash(binding),
            contract=_contract(protocol),
        )
        if value is None or value["status"] != "locked_success":
            raise LLMRelevanceJudgingIncomplete("adjudication_incomplete")
        for row in value["locked_response"]["decisions"]:
            decisions[str(row["item_id"])] = dict(row)
    records: list[dict[str, Any]] = []
    for item in items:
        one = first[item.item_id]
        two = second[item.item_id]
        if one["label"] == two["label"]:
            final_label = one["label"]
            resolution = "round_agreement"
            adjudication_evidence = None
        else:
            decision = decisions.get(item.item_id)
            if decision is None:
                raise LLMRelevanceJudgingIncomplete("adjudication_decision_missing")
            final_label = decision["final_label"]
            resolution = "llm_adjudicated"
            adjudication_evidence = decision["evidence"]
        records.append(
            {
                "item_id": item.item_id,
                "independent_1": {
                    "label": one["label"],
                    "evidence": one["evidence"],
                },
                "independent_2": {
                    "label": two["label"],
                    "evidence": two["evidence"],
                },
                "final_label": final_label,
                "resolution": resolution,
                "adjudication_evidence": adjudication_evidence,
            }
        )
    return records


def _labels_lock(
    protocol: Mapping[str, Any],
    *,
    run_dir: Path,
    prepared: Mapping[str, Any],
    runtime_binding: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch_files = sorted(
        [*run_dir.glob("rounds/*/batch_*.json"), *run_dir.glob("adjudication/batch_*.json")],
        key=lambda path: path.relative_to(run_dir).as_posix(),
    )
    inventory = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in batch_files
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "state": "labels_locked",
        "blind_view_sha256": prepared["blind_view_sha256"],
        "input_bindings_sha256": stable_hash(protocol["inputs"]),
        "runtime_binding_sha256": stable_hash(runtime_binding),
        "label_count": len(records),
        "disagreement_count": sum(
            item["independent_1"]["label"] != item["independent_2"]["label"]
            for item in records
        ),
        "adjudicated_count": sum(
            item["resolution"] == "llm_adjudicated" for item in records
        ),
        "labels_sha256": stable_hash(list(records)),
        "locked_batch_tree_sha256": stable_hash(inventory),
        "batch_inventory": inventory,
    }


def _score_locked_records(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_by_item = {str(item["item_id"]): str(item["final_label"]) for item in records}
    first_by_item = {
        str(item["item_id"]): str(item["independent_1"]["label"])
        for item in records
    }
    second_by_item = {
        str(item["item_id"]): str(item["independent_2"]["label"])
        for item in records
    }
    _, package_index = _build_blind_items(protocol, repository_root=repository_root)
    item_by_package = {
        (item["scope"], item["sample_id"]): item["item_id"]
        for item in package_index
    }
    mapping = _read_json(
        _repo_path(
            repository_root,
            protocol["inputs"]["current_package"]["mapping_path"],
        )
    )
    current_rows_one: list[dict[str, Any]] = []
    current_rows_two: list[dict[str, Any]] = []
    current_adjudication: list[dict[str, Any]] = []
    for sample in mapping["samples"]:
        sample_id = str(sample["sample_id"])
        item_id = item_by_package[("current", sample_id)]
        current_rows_one.append(
            {
                "sample_id": sample_id,
                "annotator_id": "llm-independent-1",
                "label": first_by_item[item_id],
                "notes": "",
            }
        )
        current_rows_two.append(
            {
                "sample_id": sample_id,
                "annotator_id": "llm-independent-2",
                "label": second_by_item[item_id],
                "notes": "",
            }
        )
        current_adjudication.append(
            {
                "sample_id": sample_id,
                "adjudicator_id": "llm-adjudicator",
                "final_label": (
                    final_by_item[item_id]
                    if first_by_item[item_id] != second_by_item[item_id]
                    else None
                ),
                "rationale": "",
            }
        )
    prior_labels = {
        str(overlap["prior_sample_id"]): final_by_item[
            item_by_package[("prior", str(overlap["prior_sample_id"]))]
        ]
        for overlap in mapping["prior_package_overlaps"]
    }
    existing = evaluate_full_swap_annotations(
        mapping,
        current_rows_one,
        current_rows_two,
        current_adjudication,
        prior_resolved_labels=prior_labels,
    )
    all_first = [first_by_item[item["item_id"]] for item in package_index]
    all_second = [second_by_item[item["item_id"]] for item in package_index]
    disagreement_count = sum(
        left != right for left, right in zip(all_first, all_second, strict=True)
    )
    arm_proxy, statistical = _change_only_proxy_statistics(
        protocol,
        repository_root=repository_root,
        mapping=mapping,
        item_by_package=item_by_package,
        final_by_item=final_by_item,
    )
    insufficient = sum(label == "insufficient_information" for label in final_by_item.values())
    return {
        "coverage": {
            "expected_item_count": len(package_index),
            "resolved_item_count": len(final_by_item),
            "sufficiently_informed_item_count": len(final_by_item) - insufficient,
            "insufficient_information_count": insufficient,
            "missing_item_count": 0,
            "coverage_rate": 1.0,
        },
        "agreement": {
            "item_count": len(package_index),
            "agreement_count": len(package_index) - disagreement_count,
            "disagreement_count": disagreement_count,
            "disagreement_rate": disagreement_count / len(package_index),
            "adjudicated_count": disagreement_count,
            "adjudication_rate": (
                disagreement_count / len(package_index)
            ),
            "cohen_kappa": cohen_kappa(all_first, all_second),
        },
        "change_only_arm_proxy": arm_proxy,
        "paired_statistics": statistical,
        "existing_change_only_scorer": existing,
        "absolute_precision_at_20": {
            "baseline": None,
            "candidate": None,
            "reason": "shared_top20_items_not_present_in_frozen_change_only_package",
        },
    }


def _change_only_proxy_statistics(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    mapping: Mapping[str, Any],
    item_by_package: Mapping[tuple[str, str], str],
    final_by_item: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for sample in mapping["samples"]:
        label = final_by_item[item_by_package[("current", str(sample["sample_id"]))]]
        relations.extend({**dict(item), "label": label} for item in sample["occurrences"])
    for overlap in mapping["prior_package_overlaps"]:
        sample_id = str(overlap["prior_sample_id"])
        label = final_by_item[item_by_package[("prior", sample_id)]]
        relations.extend({**dict(item), "label": label} for item in overlap["occurrences"])
    if len(relations) != 471:
        raise LLMRelevanceJudgingError("change_relation_count_mismatch")
    arms = {
        "baseline_removed": _arm_relation_proxy(relations, "baseline_removed"),
        "candidate_admitted": _arm_relation_proxy(relations, "experiment_admitted"),
    }
    case_rows = _read_jsonl(
        _repo_path(repository_root, protocol["inputs"]["record160_cases"]["path"])
    )
    included = [item for item in case_rows if item.get("included_main_analysis") is True]
    if len(included) != int(
        protocol["inputs"]["record160_cases"]["expected_query_count"]
    ):
        raise LLMRelevanceJudgingError("record160_query_count_mismatch")
    assignments = _read_jsonl(
        _repo_path(repository_root, protocol["inputs"]["cluster_assignments"]["path"])
    )
    component_by_query = {str(item["query_id"]): str(item["component_id"]) for item in assignments}
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_case[str(relation["case_id"])].append(relation)
    query_rows: list[dict[str, Any]] = []
    top_k = int(protocol["statistics"]["top_k"])
    for case in sorted(included, key=lambda item: int(item["case_order"])):
        case_id = str(case["case_id"])
        if case_id not in component_by_query:
            raise LLMRelevanceJudgingError("frozen_component_missing")
        rows = by_case.get(case_id, [])
        informed = all(item["label"] != "insufficient_information" for item in rows)
        baseline_positive = sum(
            item["direction"] == "baseline_removed"
            and item["label"] in POSITIVE_LABELS
            for item in rows
        )
        candidate_positive = sum(
            item["direction"] == "experiment_admitted"
            and item["label"] in POSITIVE_LABELS
            for item in rows
        )
        query_rows.append(
            {
                "query_identity": "query:" + hashlib.sha256(case_id.encode("utf-8")).hexdigest(),
                "component_identity": "component:" + hashlib.sha256(
                    component_by_query[case_id].encode("utf-8")
                ).hexdigest(),
                "complete_informed": informed,
                "baseline_changed_positive_contribution": baseline_positive / top_k,
                "candidate_changed_positive_contribution": candidate_positive / top_k,
                "difference": (candidate_positive - baseline_positive) / top_k,
                "relation_count": len(rows),
                "insufficient_information_count": sum(
                    item["label"] == "insufficient_information" for item in rows
                ),
            }
        )
    complete = [item for item in query_rows if item["complete_informed"]]
    query_statistics = _signal_statistics(
        complete,
        seed=int(protocol["statistics"]["query_bootstrap_seed"]),
        unit_label="query",
        iterations=int(protocol["statistics"]["bootstrap_iterations"]),
    )
    by_component: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in complete:
        by_component[str(item["component_identity"])].append(item)
    component_rows = [
        {
            "component_identity": component,
            "query_count": len(rows),
            "baseline": statistics.fmean(
                float(item["baseline_changed_positive_contribution"]) for item in rows
            ),
            "candidate": statistics.fmean(
                float(item["candidate_changed_positive_contribution"]) for item in rows
            ),
            "difference": statistics.fmean(float(item["difference"]) for item in rows),
        }
        for component, rows in sorted(by_component.items())
    ]
    cluster_statistics = _signal_statistics(
        component_rows,
        seed=int(protocol["statistics"]["cluster_bootstrap_seed"]),
        unit_label="component",
        iterations=int(protocol["statistics"]["bootstrap_iterations"]),
        baseline_key="baseline",
        candidate_key="candidate",
    )
    return arms, {
        "estimand": "change_only_positive_top20_contribution_candidate_minus_baseline",
        "all_query_count": len(query_rows),
        "complete_informed_query_count": len(complete),
        "excluded_query_count": len(query_rows) - len(complete),
        "strict_full_coverage_difference": (
            statistics.fmean(float(item["difference"]) for item in query_rows)
            if len(complete) == len(query_rows)
            else None
        ),
        "query_equal": query_statistics,
        "cluster_equal": cluster_statistics,
        "component_count": len(component_rows),
        "component_assignment": protocol["statistics"]["cluster_assignment"],
    }


def _arm_relation_proxy(
    relations: Sequence[Mapping[str, Any]], direction: str
) -> dict[str, Any]:
    rows = [item for item in relations if item["direction"] == direction]
    informed = [item for item in rows if item["label"] != "insufficient_information"]
    positive = sum(item["label"] in POSITIVE_LABELS for item in informed)
    return {
        "relation_count": len(rows),
        "sufficiently_informed_denominator": len(informed),
        "positive_numerator": positive,
        "insufficient_information_count": len(rows) - len(informed),
        "changed_item_precision_proxy": positive / len(informed) if informed else None,
    }


def _signal_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    unit_label: str,
    iterations: int,
    baseline_key: str = "baseline_changed_positive_contribution",
    candidate_key: str = "candidate_changed_positive_contribution",
) -> dict[str, Any] | None:
    if not rows:
        return None
    baseline = [float(item[baseline_key]) for item in rows]
    candidate = [float(item[candidate_key]) for item in rows]
    differences = [right - left for left, right in zip(baseline, candidate, strict=True)]
    return cluster_metric_statistics(
        differences,
        baseline,
        candidate,
        bootstrap_seed=seed,
        permutation_seed=seed,
        bootstrap_iterations=iterations,
        permutation_iterations=iterations,
        tie_tolerance=1e-12,
        unit_label=unit_label,
    )


def _publish_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    publish_dir: Path,
    records: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> None:
    _validate_publish_directory(publish_dir, repository_root)
    publish_dir.mkdir(parents=True, exist_ok=True)
    labels_path = publish_dir / "labels.jsonl"
    calls_path = publish_dir / "calls.jsonl"
    result_path = publish_dir / "result.json"
    _write_once_jsonl(labels_path, records)
    call_rows = _call_audit_rows(run_dir)
    _write_once_jsonl(calls_path, call_rows)
    _write_once_json(result_path, result)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": _contract(protocol),
        "score_scope": SCORE_SCOPE,
        "protocol_sha256": stable_hash(protocol),
        "runtime_binding": _read_json(run_dir / "runtime_binding.json"),
        "labels": {
            "path": "labels.jsonl",
            "size_bytes": labels_path.stat().st_size,
            "sha256": sha256_file(labels_path),
            "record_count": len(records),
        },
        "calls": {
            "path": "calls.jsonl",
            "size_bytes": calls_path.stat().st_size,
            "sha256": sha256_file(calls_path),
            "record_count": len(call_rows),
            "contains_prompt_or_response_content": False,
        },
        "result": {
            "path": "result.json",
            "size_bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
        },
        "source_lock_sha256": sha256_file(run_dir / "labels_lock.json"),
        "human_label_directory_modified": False,
        "default_strategy_changed": False,
    }
    _write_once_json(publish_dir / "manifest.json", manifest)


def _call_audit_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(
        [
            *run_dir.glob("rounds/*/batch_*.json"),
            *run_dir.glob("adjudication/batch_*.json"),
        ],
        key=lambda item: item.relative_to(run_dir).as_posix(),
    )
    for path in paths:
        value = _read_json(path)
        rows.append(
            {
                "mode": value["mode"],
                "phase": value["phase"],
                "batch_index": value["batch_index"],
                "batch_identity": value["batch_identity"],
                "item_count": len(value["item_ids"]),
                "status": value["status"],
                "attempts": [
                    {
                        "attempt": attempt["attempt"],
                        "status": attempt["status"],
                        "input_sha256": attempt["input_sha256"],
                        "response_sha256": attempt["response_sha256"],
                        "usage": attempt["usage"],
                        "diagnostics": attempt["diagnostics"],
                        "failure": attempt["failure"],
                    }
                    for attempt in value["attempts"]
                ],
            }
        )
    return rows


def _usage_summary(run_dir: Path) -> dict[str, Any]:
    attempts: list[Mapping[str, Any]] = []
    for path in sorted(
        [*run_dir.glob("rounds/*/batch_*.json"), *run_dir.glob("adjudication/batch_*.json")],
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        attempts.extend(_read_json(path).get("attempts") or [])
    reported = [
        item
        for item in attempts
        if item.get("usage", {}).get("status") == "supplier_reported"
    ]
    unknown = len(attempts) - len(reported)
    token_totals = {
        field: _sum_reported_usage_field(reported, field)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "logical_call_count": len(attempts),
        "successful_logical_call_count": sum(
            item.get("status") == "success" for item in attempts
        ),
        "failed_logical_call_count": sum(
            item.get("status") != "success" for item in attempts
        ),
        "known_http_attempt_count": sum(
            int(item.get("diagnostics", {}).get("http_attempts") or 0)
            for item in attempts
        ),
        "supplier_usage_reported_call_count": len(reported),
        "supplier_usage_unavailable_call_count": unknown,
        **token_totals,
        "provider_cost": {
            "status": "not_available",
            "amount": None,
            "currency": None,
        },
    }


def _sum_reported_usage_field(
    attempts: Sequence[Mapping[str, Any]], field: str
) -> int | None:
    values = [item.get("usage", {}).get(field) for item in attempts]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _last_call_usage(client: Any) -> dict[str, Any]:
    explicit = getattr(client, "last_call_usage_fields", None)
    if isinstance(explicit, Mapping):
        return {
            "status": "supplier_reported",
            "reported_fields": sorted(str(field) for field in explicit),
            "prompt_tokens": explicit.get("prompt_tokens"),
            "completion_tokens": explicit.get("completion_tokens"),
            "total_tokens": explicit.get("total_tokens"),
        }
    usage = getattr(client, "last_call_usage", None)
    if usage is None:
        return {
            "status": "not_available",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    if hasattr(usage, "model_dump"):
        values = usage.model_dump()
    elif isinstance(usage, Mapping):
        values = dict(usage)
    else:
        return {
            "status": "not_available",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "status": "supplier_reported",
        "reported_fields": [
            "completion_tokens",
            "prompt_tokens",
            "total_tokens",
        ],
        "prompt_tokens": int(values.get("prompt_tokens") or 0),
        "completion_tokens": int(values.get("completion_tokens") or 0),
        "total_tokens": int(values.get("total_tokens") or 0),
    }


def _last_call_diagnostics(client: Any) -> dict[str, Any]:
    value = getattr(client, "last_call_diagnostics", None)
    if value is None:
        return {
            "mode": "not_available",
            "http_attempts": 0,
            "latency_ms": None,
            "fallback_reason": None,
        }
    raw = value.model_dump() if hasattr(value, "model_dump") else dict(value)
    return {
        "mode": str(raw.get("mode") or "not_available"),
        "http_attempts": int(raw.get("http_attempts") or 0),
        "latency_ms": (
            int(raw["latency_ms"]) if raw.get("latency_ms") is not None else None
        ),
        "fallback_reason": (
            str(raw["fallback_reason"]) if raw.get("fallback_reason") else None
        ),
    }


def _safe_provider_failure(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    raw = details.model_dump() if hasattr(details, "model_dump") else {}
    return {
        "kind": "provider_failure",
        "error_type": type(exc).__name__,
        "http_status": raw.get("http_status"),
        "service_error_code": raw.get("service_error_code"),
    }


def _batch_identity(
    mode: str,
    phase: str,
    index: int,
    items: Sequence[BlindItem],
    *,
    contract: str,
) -> str:
    return "batch:" + stable_hash(
        {
            "contract": contract,
            "mode": mode,
            "phase": phase,
            "index": index,
            "item_ids": [item.item_id for item in items],
        }
    )


def _chunks(items: Sequence[BlindItem], size: int) -> list[list[BlindItem]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _status_report(
    status: str,
    exit_code: int,
    *,
    stage: str,
    details: Mapping[str, Any],
    contract: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": contract,
        "gate": GATE_NAME,
        "score_scope": SCORE_SCOPE,
        "status": status,
        "exit_code": exit_code,
        "stage": stage,
        "details": dict(details),
        "execution": _execution_counts(),
    }
    report["report_payload_sha256"] = stable_hash(report)
    return report


def incomplete_report(
    stage: str,
    reason: str,
    *,
    contract: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    return _status_report(
        "incomplete_or_llm_unavailable",
        EXIT_INCOMPLETE,
        stage=stage,
        details={"reason": reason},
        contract=contract,
    )


def violation_report(
    stage: str,
    reason: str,
    *,
    contract: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    return _status_report(
        "integrity_or_schema_violation",
        EXIT_INTEGRITY_VIOLATION,
        stage=stage,
        details={"reason": reason},
        contract=contract,
    )


def usage_error_report(
    reason: str = "usage_error",
    *,
    contract: str = CONTRACT_VERSION,
) -> dict[str, Any]:
    return _status_report(
        "usage_error",
        EXIT_USAGE_ERROR,
        stage="cli",
        details={"reason": reason},
        contract=contract,
    )


def _execution_counts() -> dict[str, int]:
    return {
        "academic_api_request_count": 0,
        "other_network_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
        "official_scorer_call_count": 0,
        "human_label_write_count": 0,
    }


def _validate_run_directory(run_dir: Path, repository_root: Path) -> None:
    resolved = run_dir.resolve()
    forbidden = [
        repository_root / "outputs" / "benchmark_snapshots",
        repository_root / "benchmark" / "lexical_normalization_precision_annotation",
        repository_root / "benchmark" / "lexical_normalization_record160_precision_annotation",
    ]
    for root in forbidden:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise LLMRelevanceJudgingError("run_directory_forbidden")


def _validate_publish_directory(publish_dir: Path, repository_root: Path) -> None:
    resolved = publish_dir.resolve()
    for forbidden in (
        repository_root / "benchmark" / "lexical_normalization_precision_annotation",
        repository_root
        / "benchmark"
        / "lexical_normalization_record160_precision_annotation",
        repository_root / "outputs" / "benchmark_snapshots",
    ):
        try:
            resolved.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise LLMRelevanceJudgingError("publish_directory_forbidden")


def _assert_judge_blinded_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_fields: Sequence[str],
    forbidden_fields: Sequence[str],
) -> None:
    """Validate the judge view without reusing human-package sample IDs.

    The human package validator intentionally requires ``sample_id`` values
    with a public prefix.  The LLM view replaces those IDs with irreversible
    opaque identities, so it needs the same recursive leakage check with its
    own exact public schema.
    """

    allowed = set(allowed_fields)
    forbidden = {str(field).casefold() for field in forbidden_fields}
    for row in rows:
        if set(row) != allowed:
            raise LLMRelevanceJudgingError("blind_view_field_mismatch")
        stack: list[Any] = [row]
        observed_keys: set[str] = set()
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                observed_keys.update(str(key).casefold() for key in value)
                stack.extend(value.values())
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                stack.extend(value)
        if observed_keys & forbidden:
            raise LLMRelevanceJudgingError("blind_view_forbidden_field")


def _repo_path(root: Path, value: str | Path) -> Path:
    relative = PurePosixPath(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise LLMRelevanceJudgingError("repository_path_invalid")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise LLMRelevanceJudgingError("repository_path_escape") from exc
    if not path.is_file():
        raise LLMRelevanceJudgingError("repository_input_missing")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMRelevanceJudgingError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise LLMRelevanceJudgingError("json_root_invalid")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMRelevanceJudgingError("jsonl_input_invalid") from exc
    if not all(isinstance(item, dict) for item in rows):
        raise LLMRelevanceJudgingError("jsonl_row_invalid")
    return rows


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = stable_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise LLMRelevanceJudgingError("locked_output_mismatch")
        return
    _atomic_write_bytes(path, payload)


def _write_once_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(stable_json_bytes(dict(row)) for row in rows)
    if path.exists():
        if path.read_bytes() != payload:
            raise LLMRelevanceJudgingError("locked_output_mismatch")
        return
    _atomic_write_bytes(path, payload)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, stable_json_bytes(value))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)
