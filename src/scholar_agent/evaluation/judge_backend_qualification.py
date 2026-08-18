"""Qualify an LLM backend for strict blinded-judge structured output.

The gate analyzes only sanitized historical evidence and synthetic canaries. It
never persists model labels or computes relevance-quality statistics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.evaluation.llm_relevance_judging import (
    BlindItem,
    _judge_messages,
    _opaque_item_id,
    load_protocol as load_judging_protocol,
    stable_json_bytes,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.llm.provider import LLMProviderError


CONTRACT_VERSION = "judge_backend_qualification_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "judge_backend_qualification_gate"
SCORE_SCOPE = "backend_conformance_only_not_relevance_quality_or_official_score"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
LABEL_VALUES = (
    "relevant",
    "partially_relevant",
    "not_relevant",
    "insufficient_information",
)
_CANARY_ID_PATTERN = r"^canary-[0-9]{2}$"
_OPAQUE_ITEM_PATTERN = r"^canary:[0-9a-f]{64}$"
_FORBIDDEN_PERSISTED_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "base_url",
        "endpoint",
        "headers",
        "secret",
        "response",
        "label",
        "required_label",
    }
)
_MARKDOWN_PATTERN = re.compile(
    r"```|`[^`]+`|\[[^\]]*\]\([^)]*\)|(?:^|\n)#{1,6}\s|(?:\*\*|__|\*[^*]+\*|_[^_]+_)"
)
_BOUNDARY_PATTERN = re.compile(r"[<>{}]|[\x00-\x08\x0b\x0c\x0e-\x1f\u202a-\u202e\u2066-\u2069]")


class QualificationError(RuntimeError):
    """The protocol, evidence, or probe state violates the gate contract."""


class QualificationNotEligible(QualificationError):
    """No configured backend can enter the synthetic probe."""


class JSONClient(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...


class Canary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canary_id: str = Field(pattern=_CANARY_ID_PATTERN)
    profile: str = Field(min_length=1)
    query: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str
    year: int | None
    required_label: Literal[
        "relevant",
        "partially_relevant",
        "not_relevant",
        "insufficient_information",
    ]
    expected_evidence: str = Field(min_length=1, max_length=80)


class ConformanceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=_OPAQUE_ITEM_PATTERN)
    label: Literal[
        "relevant",
        "partially_relevant",
        "not_relevant",
        "insufficient_information",
    ]
    evidence: str = Field(min_length=1, max_length=80)


class ConformanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[ConformanceLabel]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    available: bool
    reason: str | None = None
    request_options: dict[str, int | float]


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    """Load and validate the pre-registered qualification contract."""

    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("protocol_unreadable") from exc
    if not isinstance(protocol, dict):
        raise QualificationError("protocol_root_invalid")
    if protocol.get("contract") != CONTRACT_VERSION:
        raise QualificationError("protocol_contract_invalid")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise QualificationError("protocol_schema_invalid")
    if protocol.get("score_scope") != SCORE_SCOPE:
        raise QualificationError("protocol_scope_invalid")
    if protocol.get("qualification_thresholds") != {
        "fallback_count": 0,
        "item_binding_success_count": 24,
        "missing_usage_count": 0,
        "provider_failure_count": 0,
        "schema_success_count": 24,
        "strict_success_count": 24,
        "unexpected_http_retry_count": 0,
    }:
        raise QualificationError("qualification_threshold_drift")
    if protocol.get("probe") != {
        "attempts_per_canary": 1,
        "fallback_allowed": False,
        "labels_persisted": False,
        "max_candidates": 8,
        "max_concurrency": 1,
        "max_logical_calls_per_candidate": 24,
        "max_supplier_total_tokens_per_candidate": 120000,
        "native_mode": "structured_json",
        "request_options": {"max_tokens": 1024, "timeout_seconds": 60.0},
        "required_http_attempts_per_logical_call": 1,
        "required_usage_fields": [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ],
        "temperature": 0,
    }:
        raise QualificationError("probe_contract_drift")
    response_schema = protocol.get("response_schema") or {}
    if response_schema.get("labels") != list(LABEL_VALUES):
        raise QualificationError("label_enum_drift")
    if response_schema.get("malformed_output_recovery") != "forbidden":
        raise QualificationError("malformed_recovery_forbidden")

    root = repository_root.resolve()
    _validate_bound_file(root, protocol["canaries"], "path", "sha256")
    prompt = protocol.get("prompt") or {}
    _validate_bound_file(root, prompt, "system_path", "system_sha256")
    _validate_bound_file(root, prompt, "user_path", "user_sha256")
    if prompt.get("version") != "1.0.0":
        raise QualificationError("prompt_version_drift")
    for evidence in protocol.get("frozen_evidence") or []:
        for prefix in ("protocol", "calls", "manifest", "status"):
            _validate_bound_file(
                root,
                evidence,
                f"{prefix}_path",
                f"{prefix}_sha256",
            )
    canaries = load_canaries(protocol, repository_root=root)
    if len(canaries) != int(protocol["canaries"]["count"]):
        raise QualificationError("canary_count_drift")
    return protocol


def load_canaries(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> list[Canary]:
    rows = _read_jsonl(_repo_path(repository_root, protocol["canaries"]["path"]))
    try:
        canaries = [Canary.model_validate(row) for row in rows]
    except ValidationError as exc:
        raise QualificationError("canary_schema_invalid") from exc
    identifiers = [item.canary_id for item in canaries]
    if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
        raise QualificationError("canary_order_or_identity_invalid")
    expected = [f"canary-{index:02d}" for index in range(len(canaries))]
    if identifiers != expected:
        raise QualificationError("canary_sequence_drift")
    return canaries


def candidate_from_runtime(
    *,
    provider: str,
    model: str | None,
    available: bool,
    reason: str | None,
    request_options: Mapping[str, int | float],
) -> Candidate:
    """Create a safe candidate descriptor without endpoint or credential data."""

    normalized_options = {
        "max_tokens": int(request_options["max_tokens"]),
        "timeout_seconds": float(request_options["timeout_seconds"]),
    }
    identity = stable_hash(
        {
            "contract": CONTRACT_VERSION,
            "provider": provider,
            "model": model,
            "request_options": normalized_options,
        }
    )
    return Candidate(
        candidate_id=f"candidate:{identity}",
        provider=provider or "unavailable",
        model=model or "unavailable",
        available=available,
        reason=reason,
        request_options=normalized_options,
    )


def analyze_frozen_evidence(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Describe sanitized v1/v1.1 failures without recovering responses."""

    analyses = [
        _analyze_one_frozen(item, protocol, repository_root=repository_root)
        for item in protocol["frozen_evidence"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": "completed",
        "exit_code": EXIT_QUALIFIED,
        "score_scope": SCORE_SCOPE,
        "causal_interpretation": "descriptive_association_only",
        "response_content_accessed": False,
        "labels_generated_or_read": False,
        "evidence": analyses,
        "analysis_sha256": stable_hash(analyses),
        "execution": _offline_execution(),
    }


def run_probe(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    candidates: Sequence[Candidate],
    client_factory: Callable[[Candidate], JSONClient],
) -> dict[str, Any]:
    """Run every fixed canary once per candidate with resumable hash locks."""

    if len(candidates) > int(protocol["probe"]["max_candidates"]):
        raise QualificationError("candidate_limit_exceeded")
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise QualificationError("duplicate_candidate")
    canaries = load_canaries(protocol, repository_root=repository_root)
    protocol_sha256 = stable_hash(protocol)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "protocol_sha256": protocol_sha256,
        "canary_set_sha256": stable_hash(
            [item.model_dump(mode="json") for item in canaries]
        ),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "labels_persisted": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "probe_manifest.json"
    _write_or_validate_json(manifest_path, manifest)

    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.available:
            candidate_reports.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "status": "not_eligible",
                    "reason": candidate.reason or "runtime_unavailable",
                    "logical_call_count": 0,
                }
            )
            continue
        if candidate.request_options != protocol["probe"]["request_options"]:
            candidate_reports.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "provider": candidate.provider,
                    "model": candidate.model,
                    "status": "not_eligible",
                    "reason": "request_options_mismatch",
                    "logical_call_count": 0,
                }
            )
            continue
        call_path = run_dir / "calls" / f"{candidate.candidate_id.removeprefix('candidate:')}.jsonl"
        existing = _read_jsonl(call_path) if call_path.exists() else []
        by_id = _validate_existing_calls(
            existing,
            candidate=candidate,
            canaries=canaries,
            protocol_sha256=protocol_sha256,
        )
        if len(by_id) > int(protocol["probe"]["max_logical_calls_per_candidate"]):
            raise QualificationError("logical_call_limit_exceeded")
        missing_canaries = [
            canary for canary in canaries if canary.canary_id not in by_id
        ]
        client = client_factory(candidate) if missing_canaries else None
        for canary in missing_canaries:
            if len(by_id) >= int(protocol["probe"]["max_logical_calls_per_candidate"]):
                raise QualificationError("logical_call_limit_exceeded")
            assert client is not None
            row = _invoke_canary(
                protocol,
                repository_root=repository_root,
                candidate=candidate,
                canary=canary,
                client=client,
            )
            by_id[canary.canary_id] = row
            _atomic_write_jsonl(call_path, [by_id[key] for key in sorted(by_id)])
        records = [by_id[key] for key in sorted(by_id)]
        matrix = _candidate_matrix(candidate, records, protocol)
        candidate_reports.append(matrix)
    status = _probe_status(candidate_reports)
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": (
            EXIT_QUALIFIED
            if status == "qualified"
            else EXIT_VIOLATION
            if status == "conformance_violation"
            else EXIT_NOT_ELIGIBLE
        ),
        "score_scope": SCORE_SCOPE,
        "candidate_count": len(candidates),
        "candidates": candidate_reports,
        "labels_persisted": False,
        "quality_metrics_computed": False,
        "execution": {
            "academic_api_request_count": 0,
            "llm_logical_call_count": sum(
                int(item.get("logical_call_count") or 0)
                for item in candidate_reports
            ),
            "other_network_request_count": 0,
            "snapshot_write_count": 0,
        },
    }
    _atomic_write_json(run_dir / "probe_status.json", report)
    return report


def qualify_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    run_dir: Path,
    publish_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the capability matrix and optional follow-up recommendation."""

    analysis_path = run_dir / "frozen_analysis.json"
    if not analysis_path.is_file():
        raise QualificationNotEligible("frozen_analysis_missing")
    analysis = _read_json(analysis_path)
    if analysis.get("analysis_sha256") != stable_hash(analysis.get("evidence")):
        raise QualificationError("frozen_analysis_hash_mismatch")
    manifest = _read_json(run_dir / "probe_manifest.json")
    if manifest.get("protocol_sha256") != stable_hash(protocol):
        raise QualificationError("probe_protocol_mismatch")
    candidates = [Candidate.model_validate(item) for item in manifest["candidates"]]
    matrices: list[dict[str, Any]] = []
    published_calls: list[dict[str, Any]] = []
    for candidate in candidates:
        call_path = run_dir / "calls" / f"{candidate.candidate_id.removeprefix('candidate:')}.jsonl"
        records = _read_jsonl(call_path) if call_path.exists() else []
        if candidate.available and len(records) != int(protocol["canaries"]["count"]):
            raise QualificationNotEligible("probe_incomplete")
        matrices.append(_candidate_matrix(candidate, records, protocol))
        published_calls.extend(records)
    qualified = [item for item in matrices if item["qualified"]]
    recommendation = (
        _followup_manifest(protocol, qualified) if qualified else None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": "qualified" if qualified else "no_qualified_backend",
        "exit_code": EXIT_QUALIFIED if qualified else EXIT_NOT_ELIGIBLE,
        "score_scope": SCORE_SCOPE,
        "frozen_analysis_sha256": analysis["analysis_sha256"],
        "candidate_capability_matrix": matrices,
        "qualified_candidate_count": len(qualified),
        "followup_run_recommendation": recommendation,
        "relevance_labels_generated": False,
        "quality_statistics_generated": False,
        "provider_cost": {
            "amount": None,
            "currency": None,
            "status": "not_available",
        },
        "execution": {
            "academic_api_request_count": 0,
            "other_network_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }
    report["report_sha256"] = stable_hash(report)
    if publish_dir is not None:
        _publish_result(
            publish_dir,
            protocol=protocol,
            analysis=analysis,
            calls=published_calls,
            report=report,
            recommendation=recommendation,
        )
    return report


def verify_published(
    protocol: Mapping[str, Any], *, publish_dir: Path
) -> dict[str, Any]:
    manifest = _read_json(publish_dir / "manifest.json")
    if manifest.get("protocol_sha256") != stable_hash(protocol):
        raise QualificationError("published_protocol_mismatch")
    for file_info in manifest.get("files") or []:
        path = publish_dir / str(file_info["path"])
        if not path.is_file():
            raise QualificationError("published_file_missing")
        if path.stat().st_size != int(file_info["size_bytes"]):
            raise QualificationError("published_file_size_mismatch")
        if sha256_file(path) != file_info["sha256"]:
            raise QualificationError("published_file_hash_mismatch")
    report = _read_json(publish_dir / "qualification.json")
    if report.get("report_sha256") != stable_hash(
        {key: value for key, value in report.items() if key != "report_sha256"}
    ):
        raise QualificationError("published_report_hash_mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": report["status"],
        "exit_code": report["exit_code"],
        "score_scope": SCORE_SCOPE,
        "verified_file_count": len(manifest["files"]),
        "report_sha256": report["report_sha256"],
        "labels_present": False,
        "statistics_present": False,
    }


def write_frozen_analysis(run_dir: Path, report: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(run_dir / "frozen_analysis.json", report)


def _analyze_one_frozen(
    binding: Mapping[str, Any],
    qualification_protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    judging_protocol = load_judging_protocol(
        _repo_path(repository_root, binding["protocol_path"]),
        repository_root=repository_root,
    )
    status = _read_json(_repo_path(repository_root, binding["status_path"]))
    expected_ids = _expected_item_ids_from_status(status)
    items = _public_blind_items(
        judging_protocol,
        expected_ids=expected_ids,
        repository_root=repository_root,
    )
    calls = _read_jsonl(_repo_path(repository_root, binding["calls_path"]))
    attempts: list[dict[str, Any]] = []
    batch_size = int(judging_protocol["judge"]["batch_size"])
    concurrency = int(judging_protocol["judge"]["max_concurrency"])
    for call in calls:
        index = int(call["batch_index"])
        batch = items[index * batch_size : (index + 1) * batch_size]
        if len(batch) != int(call["item_count"]):
            raise QualificationError("frozen_batch_shape_mismatch")
        messages = _judge_messages(
            judging_protocol,
            repository_root=repository_root,
            items=batch,
        )
        character_count = sum(len(item["content"]) for item in messages)
        features = _batch_features(batch)
        bucket = _length_bucket(
            character_count,
            qualification_protocol["analysis"]["input_length_buckets_characters"],
        )
        for attempt in call.get("attempts") or []:
            failure = attempt.get("failure") or {}
            diagnostics = attempt.get("diagnostics") or {}
            usage = attempt.get("usage") or {}
            attempts.append(
                {
                    "attempt_number": int(attempt["attempt"]),
                    "dispatch_window": index // concurrency,
                    "features": features,
                    "http_status": failure.get("http_status"),
                    "input_length_bucket": bucket,
                    "status": attempt["status"],
                    "failure_code": (
                        failure.get("code")
                        or (
                            f"http_{failure.get('http_status')}"
                            if failure.get("http_status") is not None
                            else None
                        )
                    ),
                    "http_attempts": diagnostics.get("http_attempts"),
                    "usage_status": usage.get("status", "not_available"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            )
    return {
        "contract": binding["contract"],
        "batch_count": len(calls),
        "attempt_count": len(attempts),
        "terminal_batch_statuses": _counts(item["status"] for item in calls),
        "attempt_statuses": _counts(item["status"] for item in attempts),
        "failure_codes": _counts(
            item["failure_code"] for item in attempts if item["failure_code"]
        ),
        "http_statuses": _counts(
            str(item["http_status"])
            for item in attempts
            if item["http_status"] is not None
        ),
        "attempt_number_association": _dimension_summary(
            attempts, lambda item: str(item["attempt_number"])
        ),
        "dispatch_window_association": _dimension_summary(
            attempts, lambda item: str(item["dispatch_window"])
        ),
        "input_length_association": _dimension_summary(
            attempts, lambda item: item["input_length_bucket"]
        ),
        "text_feature_association": {
            feature: _dimension_summary(
                attempts,
                lambda item, feature=feature: str(item["features"][feature]).lower(),
            )
            for feature in ("boundary_characters", "empty_abstract", "markdown", "unicode")
        },
        "usage": _usage_summary(attempts),
        "input_feature_summary_sha256": stable_hash(
            [
                {
                    "attempt_number": item["attempt_number"],
                    "dispatch_window": item["dispatch_window"],
                    "features": item["features"],
                    "input_length_bucket": item["input_length_bucket"],
                    "status": item["status"],
                }
                for item in attempts
            ]
        ),
        "limitations": [
            "responses_are_hash_only_and_were_not_recovered",
            "dispatch_windows_are_declared_scheduler_windows_not_timestamps",
            "associations_do_not_establish_causation",
        ],
    }


def _expected_item_ids_from_status(status: Mapping[str, Any]) -> set[str]:
    round_two = status["verification"]["details"]["rounds"]["independent_2"]
    identifiers = set(str(item) for item in round_two["missing_item_ids"])
    if round_two["locked_item_count"] != 0 or len(identifiers) != int(
        round_two["expected_item_count"]
    ):
        raise QualificationError("frozen_item_identity_evidence_incomplete")
    return identifiers


def _public_blind_items(
    protocol: Mapping[str, Any],
    *,
    expected_ids: set[str],
    repository_root: Path,
) -> list[BlindItem]:
    inputs = protocol["inputs"]
    records: list[tuple[str, Mapping[str, Any], str]] = []
    for scope, package_name in (("current", "current_package"), ("prior", "prior_package")):
        package = inputs[package_name]
        for row in _read_jsonl(_repo_path(repository_root, package["public_path"])):
            item_id = _opaque_item_id(
                scope,
                str(row["sample_id"]),
                contract=str(protocol["contract"]),
                package_sha256=str(package["public_sha256"]),
            )
            if scope == "current" or item_id in expected_ids:
                records.append((item_id, row, scope))
    items = [
        BlindItem(
            item_id=item_id,
            query=str(row["query"]),
            title=str(row["title"]),
            abstract=str(row.get("abstract") or ""),
            year=row.get("year"),
        )
        for item_id, row, _scope in records
    ]
    items.sort(key=lambda item: item.item_id)
    if {item.item_id for item in items} != expected_ids:
        raise QualificationError("public_item_reconstruction_mismatch")
    return items


def _batch_features(items: Sequence[BlindItem]) -> dict[str, bool]:
    text = "\n".join(
        value
        for item in items
        for value in (item.query, item.title, item.abstract)
    )
    return {
        "boundary_characters": bool(_BOUNDARY_PATTERN.search(text)),
        "empty_abstract": any(not item.abstract for item in items),
        "markdown": bool(_MARKDOWN_PATTERN.search(text)),
        "unicode": any(ord(character) > 127 for character in text),
    }


def _length_bucket(value: int, boundaries: Sequence[int]) -> str:
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= value < upper:
            return f"{lower}-{upper - 1}"
    return f"{boundaries[-1]}+"


def _dimension_summary(
    attempts: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in attempts:
        groups[key(item)].append(item)
    return [
        {
            "value": value,
            "attempt_count": len(rows),
            "statuses": _counts(str(item["status"]) for item in rows),
        }
        for value, rows in sorted(groups.items())
    ]


def _usage_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reported = [item for item in attempts if item["usage_status"] == "supplier_reported"]
    return {
        "reported_attempt_count": len(reported),
        "not_available_attempt_count": len(attempts) - len(reported),
        "prompt_tokens": sum(int(item["prompt_tokens"] or 0) for item in reported),
        "completion_tokens": sum(
            int(item["completion_tokens"] or 0) for item in reported
        ),
        "total_tokens": sum(int(item["total_tokens"] or 0) for item in reported),
        "provider_cost": {
            "amount": None,
            "currency": None,
            "status": "not_available",
        },
    }


def _canary_item_id(protocol: Mapping[str, Any], canary: Canary) -> str:
    return "canary:" + stable_hash(
        {
            "contract": CONTRACT_VERSION,
            "protocol_sha256": stable_hash(protocol),
            "canary_id": canary.canary_id,
        }
    )


def _canary_messages(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    canary: Canary,
) -> tuple[list[dict[str, str]], str]:
    item_id = _canary_item_id(protocol, canary)
    payload = {
        "item": {
            "item_id": item_id,
            "query": canary.query,
            "title": canary.title,
            "abstract": canary.abstract,
            "year": canary.year,
            "required_label": canary.required_label,
            "expected_evidence": canary.expected_evidence,
        }
    }
    system = _repo_path(repository_root, protocol["prompt"]["system_path"]).read_text(
        encoding="utf-8"
    ).strip()
    user = _repo_path(repository_root, protocol["prompt"]["user_path"]).read_text(
        encoding="utf-8"
    ).strip()
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user.replace("{{payload}}", serialized)},
    ]
    return messages, item_id


def _invoke_canary(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    candidate: Candidate,
    canary: Canary,
    client: JSONClient,
) -> dict[str, Any]:
    messages, item_id = _canary_messages(
        protocol,
        repository_root=repository_root,
        canary=canary,
    )
    input_sha256 = stable_hash(messages)
    try:
        raw = client.chat_json(
            messages,
            temperature=float(protocol["probe"]["temperature"]),
            timeout=float(protocol["probe"]["request_options"]["timeout_seconds"]),
        )
    except Exception as exc:  # noqa: BLE001 - translated to a safe provider code
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": CONTRACT_VERSION,
            "candidate_id": candidate.candidate_id,
            "canary_id": canary.canary_id,
            "input_sha256": input_sha256,
            "response_sha256": None,
            "status": "provider_failure",
            "schema_success": False,
            "item_binding_success": False,
            "native_mode_success": False,
            "usage_complete": False,
            "failure": _safe_provider_failure(exc),
            "diagnostics": _safe_diagnostics(client),
            "usage": _safe_usage(client),
        }
    response_sha256 = stable_hash(raw)
    schema_success = False
    item_binding_success = False
    failure: dict[str, Any] | None = None
    try:
        parsed = ConformanceResponse.model_validate(raw)
        schema_success = True
        if len(parsed.labels) != 1:
            raise QualificationError("response_item_count_mismatch")
        label = parsed.labels[0]
        if label.item_id != item_id:
            raise QualificationError("response_item_binding_mismatch")
        item_binding_success = True
        if label.label != canary.required_label:
            raise QualificationError("response_required_enum_mismatch")
        if label.evidence != canary.expected_evidence:
            raise QualificationError("response_evidence_token_mismatch")
    except ValidationError:
        failure = {"kind": "schema_failure", "code": "response_schema_invalid"}
    except QualificationError as exc:
        failure = {"kind": "schema_failure", "code": str(exc)}
    diagnostics = _safe_diagnostics(client)
    usage = _safe_usage(client)
    native_mode_success = (
        diagnostics.get("mode") == protocol["probe"]["native_mode"]
        and diagnostics.get("fallback_reason") is None
        and diagnostics.get("http_attempts")
        == protocol["probe"]["required_http_attempts_per_logical_call"]
    )
    usage_complete = usage.get("status") == "supplier_reported" and set(
        usage.get("reported_fields") or []
    ) == set(protocol["probe"]["required_usage_fields"])
    strict = (
        schema_success
        and item_binding_success
        and failure is None
        and native_mode_success
        and usage_complete
    )
    if failure is None and not native_mode_success:
        failure = {"kind": "provider_conformance", "code": "native_mode_or_rate_violation"}
    if failure is None and not usage_complete:
        failure = {"kind": "usage_conformance", "code": "supplier_usage_missing"}
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "candidate_id": candidate.candidate_id,
        "canary_id": canary.canary_id,
        "input_sha256": input_sha256,
        "response_sha256": response_sha256,
        "status": "strict_success" if strict else "conformance_failure",
        "schema_success": schema_success,
        "item_binding_success": item_binding_success,
        "native_mode_success": native_mode_success,
        "usage_complete": usage_complete,
        "failure": failure,
        "diagnostics": diagnostics,
        "usage": usage,
    }


def _candidate_matrix(
    candidate: Candidate,
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(protocol["canaries"]["count"])
    provider_failures = sum(item["status"] == "provider_failure" for item in records)
    schema_successes = sum(bool(item.get("schema_success")) for item in records)
    binding_successes = sum(bool(item.get("item_binding_success")) for item in records)
    strict_successes = sum(item["status"] == "strict_success" for item in records)
    fallbacks = sum(
        (item.get("diagnostics") or {}).get("fallback_reason") is not None
        for item in records
    )
    recorded_http_attempts = [
        (item.get("diagnostics") or {}).get("http_attempts") for item in records
    ]
    retries = sum(
        value is not None
        and int(value)
        != int(protocol["probe"]["required_http_attempts_per_logical_call"])
        for value in recorded_http_attempts
    )
    attempts_unavailable = sum(value is None for value in recorded_http_attempts)
    missing_usage = sum(not bool(item.get("usage_complete")) for item in records)
    supplier_total = sum(
        int((item.get("usage") or {}).get("total_tokens") or 0)
        for item in records
        if (item.get("usage") or {}).get("status") == "supplier_reported"
    )
    counts = {
        "fallback_count": fallbacks,
        "item_binding_success_count": binding_successes,
        "missing_usage_count": missing_usage,
        "provider_failure_count": provider_failures,
        "schema_success_count": schema_successes,
        "strict_success_count": strict_successes,
        "unexpected_http_retry_count": retries,
    }
    qualified = (
        candidate.available
        and len(records) == expected
        and counts == protocol["qualification_thresholds"]
        and supplier_total
        <= int(protocol["probe"]["max_supplier_total_tokens_per_candidate"])
    )
    return {
        "candidate_id": candidate.candidate_id,
        "provider": candidate.provider,
        "model": candidate.model,
        "available": candidate.available,
        "reason": candidate.reason,
        "logical_call_count": len(records),
        "provider_success_rate": (
            (len(records) - provider_failures) / expected if expected else None
        ),
        "schema_success_rate": schema_successes / expected if expected else None,
        "strict_success_rate": strict_successes / expected if expected else None,
        "counts": counts,
        "transport_diagnostics": {
            "http_attempts_unavailable_count": attempts_unavailable,
        },
        "supplier_usage": {
            "reported_call_count": len(records) - missing_usage,
            "not_available_call_count": missing_usage,
            "prompt_tokens": sum(
                int((item.get("usage") or {}).get("prompt_tokens") or 0)
                for item in records
            ),
            "completion_tokens": sum(
                int((item.get("usage") or {}).get("completion_tokens") or 0)
                for item in records
            ),
            "total_tokens": supplier_total,
        },
        "provider_cost": {
            "amount": None,
            "currency": None,
            "status": "not_available",
        },
        "failure_codes": _counts(
            str(
                (item.get("failure") or {}).get("code")
                or (item.get("failure") or {}).get("kind")
            )
            for item in records
            if item.get("failure")
        ),
        "qualified": qualified,
    }


def _probe_status(reports: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(item.get("qualified")) for item in reports):
        return "qualified"
    if any(item.get("logical_call_count") for item in reports):
        return "conformance_violation"
    return "not_eligible"


def _followup_manifest(
    protocol: Mapping[str, Any], qualified: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "llm_relevance_judging_v1_1_followup_recommendation",
        "source_qualification_contract": CONTRACT_VERSION,
        "source_protocol_sha256": stable_hash(protocol),
        "status": "read_only_recommendation_not_executed",
        "candidate": {
            key: qualified[0][key]
            for key in ("candidate_id", "provider", "model")
        },
        "judging_protocol": "llm_relevance_judging_v1_1",
        "prompt_version": "1.1.0",
        "max_concurrency": protocol["probe"]["max_concurrency"],
        "attempts_per_item": protocol["probe"]["attempts_per_canary"],
        "request_options": dict(protocol["probe"]["request_options"]),
        "full_run_started": False,
        "score_scope": "internal_llm_proxy_not_human_or_official",
    }


def _publish_result(
    publish_dir: Path,
    *,
    protocol: Mapping[str, Any],
    analysis: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    recommendation: Mapping[str, Any] | None,
) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for name, value in (
        ("frozen_analysis.json", analysis),
        ("qualification.json", report),
    ):
        path = publish_dir / name
        _write_or_validate_json(path, value)
        files.append(path)
    calls_path = publish_dir / "calls.jsonl"
    _write_or_validate_jsonl(calls_path, calls)
    files.append(calls_path)
    if recommendation is not None:
        path = publish_dir / "followup_run_recommendation.json"
        _write_or_validate_json(path, recommendation)
        files.append(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "protocol_sha256": stable_hash(protocol),
        "state": report["status"],
        "labels_file": None,
        "statistics_file": None,
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        ],
    }
    _write_or_validate_json(publish_dir / "manifest.json", manifest)


def _validate_existing_calls(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: Candidate,
    canaries: Sequence[Canary],
    protocol_sha256: str,
) -> dict[str, dict[str, Any]]:
    del protocol_sha256
    allowed = {item.canary_id for item in canaries}
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        canary_id = str(row.get("canary_id") or "")
        if row.get("candidate_id") != candidate.candidate_id:
            raise QualificationError("cross_candidate_call_mixing")
        if row.get("contract") != CONTRACT_VERSION:
            raise QualificationError("call_contract_mismatch")
        if canary_id not in allowed:
            raise QualificationError("unknown_canary_call")
        if canary_id in output:
            raise QualificationError("duplicate_canary_attempt")
        if set(row) & _FORBIDDEN_PERSISTED_KEYS:
            raise QualificationError("call_contains_forbidden_content")
        output[canary_id] = row
    return output


def _safe_provider_failure(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    if isinstance(exc, LLMProviderError) and details is not None:
        return {
            "kind": "provider_failure",
            "error_type": type(exc).__name__,
            "http_status": getattr(details, "http_status", None),
            "service_error_code": getattr(details, "service_error_code", None),
        }
    return {
        "kind": "provider_failure",
        "error_type": type(exc).__name__,
        "http_status": None,
        "service_error_code": None,
    }


def _safe_diagnostics(client: Any) -> dict[str, Any]:
    value = getattr(client, "last_call_diagnostics", None)
    if value is None:
        return {
            "mode": None,
            "http_attempts": None,
            "latency_ms": None,
            "fallback_reason": None,
        }
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        return {
            "mode": None,
            "http_attempts": None,
            "latency_ms": None,
            "fallback_reason": None,
        }
    return {
        "mode": value.get("mode"),
        "http_attempts": value.get("http_attempts"),
        "latency_ms": value.get("latency_ms"),
        "fallback_reason": value.get("fallback_reason"),
    }


def _safe_usage(client: Any) -> dict[str, Any]:
    value = getattr(client, "last_call_usage_fields", None)
    if not isinstance(value, Mapping):
        return {
            "status": "not_available",
            "reported_fields": [],
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    fields = sorted(
        key
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(value.get(key), int)
    )
    return {
        "status": "supplier_reported" if fields else "not_available",
        "reported_fields": fields,
        "prompt_tokens": value.get("prompt_tokens"),
        "completion_tokens": value.get("completion_tokens"),
        "total_tokens": value.get("total_tokens"),
    }


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _offline_execution() -> dict[str, int]:
    return {
        "academic_api_request_count": 0,
        "llm_logical_call_count": 0,
        "other_network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }


def _validate_bound_file(
    root: Path,
    binding: Mapping[str, Any],
    path_key: str,
    hash_key: str,
) -> None:
    path = _repo_path(root, str(binding.get(path_key) or ""))
    if not path.is_file() or sha256_file(path) != binding.get(hash_key):
        raise QualificationError(f"bound_file_mismatch:{path_key}")


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise QualificationError("path_outside_repository")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError("json_root_invalid")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise QualificationError("jsonl_row_invalid")
            rows.append(value)
    return rows


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, stable_json_bytes(value))


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(stable_json_bytes(row) for row in rows)
    _atomic_write_bytes(path, payload)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = stable_json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise QualificationError("locked_json_mismatch")
        return
    _atomic_write_bytes(path, payload)


def _write_or_validate_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    payload = b"".join(stable_json_bytes(row) for row in rows)
    if path.exists():
        if path.read_bytes() != payload:
            raise QualificationError("locked_jsonl_mismatch")
        return
    _atomic_write_bytes(path, payload)
