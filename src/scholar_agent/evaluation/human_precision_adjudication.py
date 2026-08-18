"""Offline integrity and adjudication gate for frozen blind-label packages.

The gate accepts only opaque package item identities.  It validates the
tracked package and frozen rubric before importing independent labels, then
delegates completed statistics to the existing full-swap scorer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.evaluation.execution_determinism import (
    forbid_network,
    tree_signature,
)
from scholar_agent.evaluation.full_swap_precision_annotation import (
    evaluate_full_swap_annotations,
)
from scholar_agent.evaluation.precision_annotation import (
    LABELS,
    PUBLIC_FIELDS,
    assert_blinded_rows,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


CONTRACT_VERSION = "human_precision_adjudication_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "human_precision_adjudication_gate"
EXIT_VALIDATED = 0
EXIT_INTEGRITY_VIOLATION = 2
EXIT_PENDING_OR_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
DEFAULT_SNAPSHOT_ROOT = (
    Path(__file__).resolve().parents[3] / "outputs" / "benchmark_snapshots"
)
ANNOTATOR_ID_PATTERN = r"^anon-[A-Za-z0-9_-]{1,64}$"

GateState = Literal[
    "awaiting_labels",
    "adjudication_required",
    "validated",
    "invalid",
    "not_eligible",
]


class HumanPrecisionGateError(RuntimeError):
    """Malformed CLI input or protocol usage."""


class PackageNotEligible(HumanPrecisionGateError):
    """The frozen package or rubric cannot safely enter label import."""


class LabelIntegrityViolation(HumanPrecisionGateError):
    """A submission violates the blinded label contract."""

    def __init__(
        self,
        code: str,
        *,
        path: str = "$",
        item_identity: str | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.path = path
        self.item_identity_sha256 = (
            _opaque_digest(item_identity) if item_identity else None
        )


class PackageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_id: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LabelRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    label: str
    notes: str | None = None


class IndependentSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    contract: Literal["human_precision_adjudication_v1"]
    package: PackageReference
    round_id: Literal["independent_1", "independent_2"]
    annotator_id: str = Field(pattern=ANNOTATOR_ID_PATTERN)
    labels: list[LabelRow]


class AdjudicationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    final_label: str
    rationale: str | None = None


class AdjudicationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    contract: Literal["human_precision_adjudication_v1"]
    package: PackageReference
    round_id: Literal["adjudication"]
    adjudicator_id: str = Field(pattern=ANNOTATOR_ID_PATTERN)
    decisions: list[AdjudicationRow]


class PriorLabelRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    annotator_1_label: str
    annotator_2_label: str
    final_label: str
    resolution: Literal["annotator_agreement", "adjudicated"]


class PriorResolvedSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    contract: Literal["human_precision_adjudication_v1"]
    package: PackageReference
    round_id: Literal["resolved_prior_package"]
    annotator_1_id: str = Field(pattern=ANNOTATOR_ID_PATTERN)
    annotator_2_id: str = Field(pattern=ANNOTATOR_ID_PATTERN)
    adjudicator_id: str | None = Field(
        default=None,
        pattern=ANNOTATOR_ID_PATTERN,
    )
    labels: list[PriorLabelRow]


class PackageContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    reference: PackageReference
    mapping: dict[str, Any]
    item_ids: list[str]
    prior_reference: PackageReference | None = None
    required_prior_item_ids: list[str] = Field(default_factory=list)
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HumanPrecisionGateError("protocol_unreadable") from exc
    if not isinstance(protocol, dict):
        raise HumanPrecisionGateError("protocol_root_must_be_object")
    if (
        protocol.get("schema_version") != SCHEMA_VERSION
        or protocol.get("contract") != CONTRACT_VERSION
    ):
        raise HumanPrecisionGateError("protocol_version_incompatible")
    if protocol.get("score_scope") != "internal_non_official_human_precision":
        raise HumanPrecisionGateError("protocol_score_scope_invalid")
    if protocol.get("digest_algorithms") != {
        "opaque_item_identity": "sha256_utf8_v1",
        "package_tree": "sorted_relative_path_size_sha256_v1",
    }:
        raise HumanPrecisionGateError("protocol_digest_algorithm_invalid")
    annotation = protocol.get("annotation")
    if not isinstance(annotation, dict):
        raise HumanPrecisionGateError("annotation_contract_missing")
    if tuple(annotation.get("labels") or []) != LABELS:
        raise HumanPrecisionGateError("protocol_labels_drift")
    if annotation.get("independent_annotator_count") != 2:
        raise HumanPrecisionGateError("independent_annotator_count_invalid")
    if annotation.get("adjudication_required_on_disagreement") is not True:
        raise HumanPrecisionGateError("adjudication_semantics_invalid")
    if annotation.get("confidence_supported") is not False:
        raise HumanPrecisionGateError("unregistered_confidence_semantics")
    if annotation.get("annotator_identity_format") != "opaque_anon_identifier_v1":
        raise HumanPrecisionGateError("annotator_identity_format_invalid")
    if annotation.get("annotation_optional_fields") != ["notes"]:
        raise HumanPrecisionGateError("annotation_optional_fields_invalid")
    if annotation.get("adjudication_optional_fields") != ["rationale"]:
        raise HumanPrecisionGateError("adjudication_optional_fields_invalid")
    if protocol.get("exclusions") != {
        "allowed_reasons": [],
        "expected_count": 0,
    }:
        raise HumanPrecisionGateError("exclusion_semantics_unregistered")
    evaluator = protocol.get("evaluator") or {}
    if (
        evaluator.get("identity_version") != "deduplicated_gold_identity_v2"
        or evaluator.get("statistics_version")
        != "full_swap_precision_annotation_v1"
    ):
        raise HumanPrecisionGateError("evaluator_version_invalid")
    _validate_package_binding_shape(protocol.get("package"), "package")
    prior = protocol.get("prior_package")
    if prior is not None:
        _validate_package_binding_shape(prior, "prior_package")
    return protocol


def validate_package(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> PackageContext:
    binding = protocol["package"]
    root = _repo_path(repository_root, str(binding["root"]))
    _validate_package_files(root, binding)
    manifest = _read_json(_bound_file(root, binding, "manifest_path"))
    rubric_path = _bound_file(root, binding, "rubric_path")
    rubric = _read_json(rubric_path)
    mapping = _read_json(_bound_file(root, binding, "mapping_path"))
    public_rows = _read_jsonl(_bound_file(root, binding, "public_samples_path"))
    annotator_one = _read_jsonl(
        _bound_file(root, binding, "annotator_1_template_path")
    )
    annotator_two = _read_jsonl(
        _bound_file(root, binding, "annotator_2_template_path")
    )
    adjudication = _read_jsonl(
        _bound_file(root, binding, "adjudication_template_path")
    )

    if manifest.get("package") != binding["package_id"]:
        raise PackageNotEligible("package_manifest_identity_mismatch")
    if str(manifest.get("version")) != str(binding["package_version"]):
        raise PackageNotEligible("package_manifest_version_mismatch")
    if mapping.get("package") != binding["package_id"]:
        raise PackageNotEligible("private_mapping_identity_mismatch")
    _validate_rubric(rubric, manifest, protocol)
    try:
        assert_blinded_rows(
            public_rows, manifest.get("forbidden_public_fields") or []
        )
    except ValueError as exc:
        raise PackageNotEligible("blind_package_schema_invalid") from exc
    item_ids = _unique_ids(mapping.get("samples") or [], "sample_id", "mapping")
    expected_count = int(binding["expected_item_count"])
    if len(item_ids) != expected_count:
        raise PackageNotEligible("package_item_count_drift")
    if stable_hash(sorted(item_ids)) != binding["item_set_sha256"]:
        raise PackageNotEligible("package_item_identity_drift")
    for name, rows in (
        ("public_samples", public_rows),
        ("annotator_1_template", annotator_one),
        ("annotator_2_template", annotator_two),
        ("adjudication_template", adjudication),
    ):
        row_ids = _unique_ids(rows, "sample_id", name)
        if set(row_ids) != set(item_ids):
            raise PackageNotEligible(f"{name}_coverage_drift")
    if any(row.get("label") is not None for row in [*annotator_one, *annotator_two]):
        raise PackageNotEligible("frozen_template_contains_human_labels")
    if any(row.get("final_label") is not None for row in adjudication):
        raise PackageNotEligible("frozen_adjudication_contains_human_labels")

    required_prior_ids = sorted(
        str(item["prior_sample_id"])
        for item in mapping.get("prior_package_overlaps") or []
    )
    prior_binding = protocol.get("prior_package")
    prior_reference: PackageReference | None = None
    if required_prior_ids:
        if not isinstance(prior_binding, Mapping):
            raise PackageNotEligible("prior_package_binding_missing")
        prior_root = _repo_path(repository_root, str(prior_binding["root"]))
        _validate_package_files(prior_root, prior_binding)
        prior_manifest = _read_json(
            _bound_file(prior_root, prior_binding, "manifest_path")
        )
        prior_rubric = _read_json(
            _bound_file(prior_root, prior_binding, "rubric_path")
        )
        prior_mapping = _read_json(
            _bound_file(prior_root, prior_binding, "mapping_path")
        )
        if prior_manifest.get("package") != prior_binding["package_id"]:
            raise PackageNotEligible("prior_manifest_identity_mismatch")
        if _manifest_version(prior_manifest) != str(
            prior_binding["package_version"]
        ):
            raise PackageNotEligible("prior_manifest_version_mismatch")
        if prior_mapping.get("package") != prior_binding["package_id"]:
            raise PackageNotEligible("prior_mapping_identity_mismatch")
        _validate_rubric_labels(prior_rubric)
        prior_item_ids = sorted(
            _unique_ids(prior_mapping.get("samples") or [], "sample_id", "prior")
        )
        if len(prior_item_ids) != int(prior_binding["expected_item_count"]):
            raise PackageNotEligible("prior_item_count_drift")
        if stable_hash(prior_item_ids) != prior_binding["item_set_sha256"]:
            raise PackageNotEligible("prior_item_identity_drift")
        if not set(required_prior_ids) <= set(prior_item_ids):
            raise PackageNotEligible("prior_item_reference_missing")
        if len(required_prior_ids) != int(binding["required_prior_item_count"]):
            raise PackageNotEligible("required_prior_item_count_drift")
        if (
            stable_hash(required_prior_ids)
            != binding["required_prior_item_set_sha256"]
        ):
            raise PackageNotEligible("required_prior_item_identity_drift")
        prior_reference = _package_reference(prior_binding)
    elif int(binding.get("required_prior_item_count", 0)) != 0:
        raise PackageNotEligible("unexpected_prior_item_contract")

    return PackageContext(
        reference=_package_reference(binding),
        mapping=mapping,
        item_ids=sorted(item_ids),
        prior_reference=prior_reference,
        required_prior_item_ids=required_prior_ids,
        rubric_sha256=sha256_file(rubric_path),
    )


def run_human_precision_gate(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    annotator_one_path: Path | None = None,
    annotator_two_path: Path | None = None,
    adjudication_path: Path | None = None,
    prior_resolved_path: Path | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    attempts = {"network": 0}
    snapshot_before = tree_signature(snapshot_root)
    with forbid_network(attempts):
        context = validate_package(protocol, repository_root=repository_root)
        first = _load_independent_submission(
            annotator_one_path,
            expected_round="independent_1",
            context=context,
            protocol=protocol,
        )
        second = _load_independent_submission(
            annotator_two_path,
            expected_round="independent_2",
            context=context,
            protocol=protocol,
        )
        adjudication = _load_adjudication_submission(
            adjudication_path,
            context=context,
            protocol=protocol,
        )
        prior = _load_prior_submission(
            prior_resolved_path,
            context=context,
            protocol=protocol,
        )
        report = _evaluate_state(
            protocol,
            context,
            first=first,
            second=second,
            adjudication=adjudication,
            prior=prior,
        )
    snapshot_after = tree_signature(snapshot_root)
    if attempts["network"]:
        raise LabelIntegrityViolation("network_access_attempted", path="$.execution")
    if snapshot_before != snapshot_after:
        raise LabelIntegrityViolation("snapshot_tree_modified", path="$.execution")
    report["input_artifacts"] = {
        "adjudication": _submission_artifact(adjudication_path),
        "independent_1": _submission_artifact(annotator_one_path),
        "independent_2": _submission_artifact(annotator_two_path),
        "prior_resolved": _submission_artifact(prior_resolved_path),
    }
    report["execution"] = {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "official_scorer_call_count": 0,
    }
    return report


def invalid_report(exc: LabelIntegrityViolation) -> dict[str, Any]:
    violation = {
        "code": exc.code,
        "path": exc.path,
        "item_identity_sha256": exc.item_identity_sha256,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "state": "invalid",
        "exit_code": EXIT_INTEGRITY_VIOLATION,
        "score_scope": "internal_non_official_human_precision",
        "statistics": None,
        "violation_count": 1,
        "violations": [violation],
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "official_scorer_call_count": 0,
        },
    }


def not_eligible_report(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "state": "not_eligible",
        "exit_code": EXIT_PENDING_OR_NOT_ELIGIBLE,
        "score_scope": "internal_non_official_human_precision",
        "reason": reason,
        "statistics": None,
        "violation_count": 0,
        "violations": [],
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "official_scorer_call_count": 0,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _evaluate_state(
    protocol: Mapping[str, Any],
    context: PackageContext,
    *,
    first: IndependentSubmission | None,
    second: IndependentSubmission | None,
    adjudication: AdjudicationSubmission | None,
    prior: PriorResolvedSubmission | None,
) -> dict[str, Any]:
    expected = set(context.item_ids)
    first_rows = _label_index(first.labels, expected) if first else {}
    second_rows = _label_index(second.labels, expected) if second else {}
    if first and second and first.annotator_id == second.annotator_id:
        raise LabelIntegrityViolation(
            "annotator_identity_reused",
            path="$.annotator_id",
        )
    received = {
        "independent_1": len(first_rows),
        "independent_2": len(second_rows),
        "adjudication": len(adjudication.decisions) if adjudication else 0,
        "prior_resolved": len(prior.labels) if prior else 0,
    }
    base = _base_report(protocol, context, received)
    base["coverage"]["missing_item_identity_sha256"] = {
        "independent_1": _missing_identity_digests(expected, set(first_rows)),
        "independent_2": _missing_identity_digests(expected, set(second_rows)),
        "prior_resolved": _missing_identity_digests(
            set(context.required_prior_item_ids),
            {row.item_id for row in prior.labels} if prior else set(),
        ),
    }
    if set(first_rows) != expected or set(second_rows) != expected:
        if adjudication and adjudication.decisions:
            raise LabelIntegrityViolation(
                "adjudication_before_independent_coverage",
                path="$.adjudication",
            )
        return {
            **base,
            "state": "awaiting_labels",
            "exit_code": EXIT_PENDING_OR_NOT_ELIGIBLE,
            "reason": "two_complete_independent_label_rounds_required",
            "agreement": None,
            "adjudication_trace": None,
            "statistics": None,
        }

    disagreements = sorted(
        item_id
        for item_id in context.item_ids
        if first_rows[item_id] != second_rows[item_id]
    )
    decisions = _adjudication_index(adjudication, expected)
    agreement_ids = expected - set(disagreements)
    forged = sorted(set(decisions) & agreement_ids)
    if forged:
        raise LabelIntegrityViolation(
            "adjudication_without_disagreement",
            path="$.adjudication.decisions",
            item_identity=forged[0],
        )
    if adjudication and adjudication.adjudicator_id in {
        first.annotator_id,
        second.annotator_id,
    }:
        raise LabelIntegrityViolation(
            "adjudicator_not_independent",
            path="$.adjudication.adjudicator_id",
        )
    missing_decisions = sorted(set(disagreements) - set(decisions))
    score_rows_one = _scorer_annotation_rows(first)
    score_rows_two = _scorer_annotation_rows(second)
    score_adjudication = [
        {
            "sample_id": item_id,
            "adjudicator_id": (
                adjudication.adjudicator_id if adjudication else ""
            ),
            "final_label": decisions.get(item_id),
            "rationale": "",
        }
        for item_id in context.item_ids
    ]
    scorer_pending = evaluate_full_swap_annotations(
        context.mapping,
        score_rows_one,
        score_rows_two,
        score_adjudication,
    )
    agreement = copy.deepcopy(scorer_pending.get("agreement"))
    traces = _decision_traces(
        context.item_ids,
        first,
        second,
        decisions,
    )
    if missing_decisions:
        return {
            **base,
            "state": "adjudication_required",
            "exit_code": EXIT_PENDING_OR_NOT_ELIGIBLE,
            "reason": "all_disagreements_require_adjudication",
            "agreement": agreement,
            "adjudication_trace": {
                "resolved_disagreement_count": len(decisions),
                "unresolved_disagreement_count": len(missing_decisions),
                "decision_records": traces,
                "trace_sha256": stable_hash(traces),
            },
            "statistics": None,
        }

    required_prior = set(context.required_prior_item_ids)
    prior_rows = _prior_label_index(prior, required_prior)
    if set(prior_rows) != required_prior:
        return {
            **base,
            "state": "awaiting_labels",
            "exit_code": EXIT_PENDING_OR_NOT_ELIGIBLE,
            "reason": "resolved_prior_package_labels_required",
            "agreement": agreement,
            "adjudication_trace": {
                "resolved_disagreement_count": len(decisions),
                "unresolved_disagreement_count": 0,
                "decision_records": traces,
                "trace_sha256": stable_hash(traces),
            },
            "statistics": None,
        }

    scored = evaluate_full_swap_annotations(
        context.mapping,
        score_rows_one,
        score_rows_two,
        score_adjudication,
        prior_resolved_labels=prior_rows,
    )
    if scored.get("annotation_status") != "complete":
        raise LabelIntegrityViolation(
            "existing_scorer_did_not_complete",
            path="$.statistics",
        )
    case_count = int(context.mapping["case_count"])
    top_k = int(context.mapping["top_k"])
    statistics = {
        "scope": "internal_non_official_change_only_precision",
        "evaluator": copy.deepcopy(protocol["evaluator"]),
        "sample_count": len(context.item_ids),
        "excluded_item_count": 0,
        "prior_resolved_item_count": len(prior_rows),
        "case_count": case_count,
        "top_k": top_k,
        "changed_component_denominator": case_count * top_k,
        "agreement": scored["agreement"],
        "metrics": scored["metrics"],
    }
    return {
        **base,
        "state": "validated",
        "exit_code": EXIT_VALIDATED,
        "reason": None,
        "agreement": scored["agreement"],
        "adjudication_trace": {
            "resolved_disagreement_count": len(decisions),
            "unresolved_disagreement_count": 0,
            "decision_records": traces,
            "trace_sha256": stable_hash(traces),
        },
        "statistics": statistics,
    }


def _base_report(
    protocol: Mapping[str, Any],
    context: PackageContext,
    received: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "score_scope": "internal_non_official_human_precision",
        "package": {
            "package_id": context.reference.package_id,
            "package_version": context.reference.package_version,
            "package_sha256": context.reference.package_sha256,
            "expected_item_count": len(context.item_ids),
            "expected_item_set_sha256": stable_hash(context.item_ids),
            "required_prior_item_count": len(context.required_prior_item_ids),
            "rubric_sha256": context.rubric_sha256,
            "identity_version": protocol["evaluator"]["identity_version"],
        },
        "coverage": {
            "expected_independent_item_count": len(context.item_ids),
            "expected_prior_resolved_item_count": len(
                context.required_prior_item_ids
            ),
            "received_counts": dict(sorted(received.items())),
            "exclusion_count": 0,
        },
        "violation_count": 0,
        "violations": [],
    }


def _load_independent_submission(
    path: Path | None,
    *,
    expected_round: str,
    context: PackageContext,
    protocol: Mapping[str, Any],
) -> IndependentSubmission | None:
    if path is None:
        return None
    payload = _read_submission(path)
    try:
        submission = IndependentSubmission.model_validate(payload)
    except ValidationError as exc:
        raise LabelIntegrityViolation(
            "independent_submission_schema_invalid",
            path="$.labels",
        ) from exc
    if submission.round_id != expected_round:
        raise LabelIntegrityViolation("independent_round_mismatch", path="$.round_id")
    _validate_reference(submission.package, context.reference)
    _validate_labels(submission.labels, protocol)
    _ensure_unique_submission_ids(submission.labels)
    _ensure_known_submission_ids(
        submission.labels,
        set(context.item_ids),
        code="unknown_item_identity",
        path="$.labels",
    )
    return submission


def _load_adjudication_submission(
    path: Path | None,
    *,
    context: PackageContext,
    protocol: Mapping[str, Any],
) -> AdjudicationSubmission | None:
    if path is None:
        return None
    payload = _read_submission(path)
    try:
        submission = AdjudicationSubmission.model_validate(payload)
    except ValidationError as exc:
        raise LabelIntegrityViolation(
            "adjudication_submission_schema_invalid",
            path="$.decisions",
        ) from exc
    _validate_reference(submission.package, context.reference)
    _validate_labels(submission.decisions, protocol, field="final_label")
    _ensure_unique_submission_ids(submission.decisions)
    _ensure_known_submission_ids(
        submission.decisions,
        set(context.item_ids),
        code="unknown_adjudication_item_identity",
        path="$.decisions",
    )
    return submission


def _load_prior_submission(
    path: Path | None,
    *,
    context: PackageContext,
    protocol: Mapping[str, Any],
) -> PriorResolvedSubmission | None:
    if path is None:
        return None
    if context.prior_reference is None:
        raise LabelIntegrityViolation("unexpected_prior_package_submission")
    payload = _read_submission(path)
    try:
        submission = PriorResolvedSubmission.model_validate(payload)
    except ValidationError as exc:
        raise LabelIntegrityViolation(
            "prior_submission_schema_invalid",
            path="$.labels",
        ) from exc
    _validate_reference(submission.package, context.prior_reference)
    _validate_prior_resolution(submission, protocol)
    _ensure_unique_submission_ids(submission.labels)
    _ensure_known_submission_ids(
        submission.labels,
        set(context.required_prior_item_ids),
        code="unknown_prior_item_identity",
        path="$.prior_labels",
    )
    return submission


def _read_submission(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelIntegrityViolation("submission_unreadable") from exc
    if not isinstance(payload, dict):
        raise LabelIntegrityViolation("submission_root_must_be_object")
    return payload


def _submission_artifact(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "sha256": None}
    return {"present": True, "sha256": sha256_file(path)}


def _validate_reference(
    observed: PackageReference, expected: PackageReference
) -> None:
    if observed != expected:
        raise LabelIntegrityViolation("submission_package_mismatch", path="$.package")


def _validate_labels(
    rows: Sequence[Any],
    protocol: Mapping[str, Any],
    *,
    field: str = "label",
) -> None:
    allowed = set(protocol["annotation"]["labels"])
    for row in rows:
        value = str(getattr(row, field))
        if value not in allowed:
            raise LabelIntegrityViolation(
                "invalid_annotation_label",
                path=f"$.{field}",
                item_identity=str(row.item_id),
            )


def _validate_prior_resolution(
    submission: PriorResolvedSubmission,
    protocol: Mapping[str, Any],
) -> None:
    allowed = set(protocol["annotation"]["labels"])
    if submission.annotator_1_id == submission.annotator_2_id:
        raise LabelIntegrityViolation(
            "prior_annotator_identity_reused",
            path="$.annotator_1_id",
        )
    adjudicated = False
    for row in submission.labels:
        values = {
            row.annotator_1_label,
            row.annotator_2_label,
            row.final_label,
        }
        if not values <= allowed:
            raise LabelIntegrityViolation(
                "invalid_annotation_label",
                path="$.prior_labels",
                item_identity=row.item_id,
            )
        agrees = row.annotator_1_label == row.annotator_2_label
        if agrees and (
            row.resolution != "annotator_agreement"
            or row.final_label != row.annotator_1_label
        ):
            raise LabelIntegrityViolation(
                "prior_resolution_trace_invalid",
                path="$.prior_labels",
                item_identity=row.item_id,
            )
        if not agrees:
            adjudicated = True
            if row.resolution != "adjudicated":
                raise LabelIntegrityViolation(
                    "prior_disagreement_not_adjudicated",
                    path="$.prior_labels",
                    item_identity=row.item_id,
                )
    if adjudicated and (
        not submission.adjudicator_id
        or submission.adjudicator_id
        in {submission.annotator_1_id, submission.annotator_2_id}
    ):
        raise LabelIntegrityViolation(
            "prior_adjudicator_not_independent",
            path="$.adjudicator_id",
        )


def _ensure_unique_submission_ids(rows: Sequence[Any]) -> None:
    observed: set[str] = set()
    for row in rows:
        item_id = str(row.item_id)
        if item_id in observed:
            raise LabelIntegrityViolation(
                "duplicate_item_submission",
                path="$.items",
                item_identity=item_id,
            )
        observed.add(item_id)


def _ensure_known_submission_ids(
    rows: Sequence[Any],
    expected: set[str],
    *,
    code: str,
    path: str,
) -> None:
    for row in rows:
        if row.item_id not in expected:
            raise LabelIntegrityViolation(
                code,
                path=path,
                item_identity=str(row.item_id),
            )


def _label_index(
    rows: Sequence[LabelRow], expected: set[str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in rows:
        if row.item_id not in expected:
            raise LabelIntegrityViolation(
                "unknown_item_identity",
                path="$.labels",
                item_identity=row.item_id,
            )
        values[row.item_id] = row.label
    return values


def _adjudication_index(
    submission: AdjudicationSubmission | None, expected: set[str]
) -> dict[str, str]:
    if submission is None:
        return {}
    values: dict[str, str] = {}
    for row in submission.decisions:
        if row.item_id not in expected:
            raise LabelIntegrityViolation(
                "unknown_adjudication_item_identity",
                path="$.decisions",
                item_identity=row.item_id,
            )
        values[row.item_id] = row.final_label
    return values


def _prior_label_index(
    submission: PriorResolvedSubmission | None, expected: set[str]
) -> dict[str, str]:
    if submission is None:
        return {}
    values: dict[str, str] = {}
    for row in submission.labels:
        if row.item_id not in expected:
            raise LabelIntegrityViolation(
                "unknown_prior_item_identity",
                path="$.prior_labels",
                item_identity=row.item_id,
            )
        values[row.item_id] = row.final_label
    return values


def _scorer_annotation_rows(
    submission: IndependentSubmission,
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row.item_id,
            "annotator_id": submission.annotator_id,
            "label": row.label,
            "notes": "",
        }
        for row in sorted(submission.labels, key=lambda item: item.item_id)
    ]


def _decision_traces(
    item_ids: Sequence[str],
    first: IndependentSubmission,
    second: IndependentSubmission,
    decisions: Mapping[str, str],
) -> list[dict[str, Any]]:
    first_rows = {row.item_id: row.label for row in first.labels}
    second_rows = {row.item_id: row.label for row in second.labels}
    traces = []
    for item_id in sorted(item_ids):
        if first_rows[item_id] == second_rows[item_id]:
            final_label = first_rows[item_id]
            resolution = "annotator_agreement"
        else:
            final_label = decisions.get(item_id)
            resolution = (
                "adjudicated" if final_label is not None else "unresolved"
            )
        traces.append(
            {
                "item_identity_sha256": _opaque_digest(item_id),
                "annotator_1_label": first_rows[item_id],
                "annotator_2_label": second_rows[item_id],
                "final_label": final_label,
                "resolution": resolution,
            }
        )
    return traces


def _validate_package_binding_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise HumanPrecisionGateError(f"{label}_binding_missing")
    required = {
        "package_id",
        "package_version",
        "root",
        "package_sha256",
        "file_count",
        "expected_item_count",
        "item_set_sha256",
        "manifest_path",
        "rubric_path",
        "mapping_path",
        "public_samples_path",
        "annotator_1_template_path",
        "annotator_2_template_path",
        "adjudication_template_path",
    }
    if not required <= set(value):
        raise HumanPrecisionGateError(f"{label}_binding_incomplete")


def _validate_package_files(root: Path, binding: Mapping[str, Any]) -> None:
    if not root.is_dir():
        raise PackageNotEligible("package_directory_missing")
    entries = _package_entries(root)
    if len(entries) != int(binding["file_count"]):
        raise PackageNotEligible("package_file_count_drift")
    if stable_hash(entries) != binding["package_sha256"]:
        raise PackageNotEligible("package_digest_mismatch")
    for field in (
        "manifest_path",
        "rubric_path",
        "mapping_path",
        "public_samples_path",
        "annotator_1_template_path",
        "annotator_2_template_path",
        "adjudication_template_path",
    ):
        path = _bound_file(root, binding, field)
        if not path.is_file():
            raise PackageNotEligible(f"package_required_file_missing:{field}")


def _validate_rubric(
    rubric: Mapping[str, Any],
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    labels = tuple(rubric.get("labels") or [])
    definitions = rubric.get("definitions") or {}
    if labels != LABELS or tuple(protocol["annotation"]["labels"]) != labels:
        raise PackageNotEligible("rubric_labels_incomplete")
    if any(not str(definitions.get(label) or "").strip() for label in labels):
        raise PackageNotEligible("rubric_definitions_incomplete")
    if rubric.get("independence") != "annotators work independently before adjudication":
        raise PackageNotEligible("rubric_independence_incomplete")
    if rubric.get("adjudication") != "required only where the two labels differ":
        raise PackageNotEligible("rubric_adjudication_incomplete")
    if set(rubric.get("public_sample_fields") or []) != set(PUBLIC_FIELDS):
        raise PackageNotEligible("rubric_public_fields_drift")
    if tuple(manifest.get("annotation", {}).get("labels") or []) != labels:
        raise PackageNotEligible("manifest_rubric_label_drift")


def _validate_rubric_labels(rubric: Mapping[str, Any]) -> None:
    labels = tuple(rubric.get("labels") or [])
    definitions = rubric.get("definitions") or {}
    if labels != LABELS:
        raise PackageNotEligible("prior_rubric_labels_incomplete")
    if any(not str(definitions.get(label) or "").strip() for label in labels):
        raise PackageNotEligible("prior_rubric_definitions_incomplete")


def _package_reference(binding: Mapping[str, Any]) -> PackageReference:
    return PackageReference(
        package_id=str(binding["package_id"]),
        package_version=str(binding["package_version"]),
        package_sha256=str(binding["package_sha256"]),
    )


def _package_entries(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _unique_ids(
    rows: Sequence[Mapping[str, Any]], field: str, label: str
) -> list[str]:
    values = [str(row.get(field) or "") for row in rows]
    if any(not item for item in values) or len(values) != len(set(values)):
        raise PackageNotEligible(f"{label}_item_identity_invalid")
    return values


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageNotEligible("package_json_unreadable") from exc
    if not isinstance(value, dict):
        raise PackageNotEligible("package_json_root_invalid")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageNotEligible("package_jsonl_unreadable") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise PackageNotEligible("package_jsonl_row_invalid")
    return rows


def _repo_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise HumanPrecisionGateError("path_must_be_repository_relative")
    root = repository_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HumanPrecisionGateError("path_resolves_outside_repository") from exc
    return resolved


def _bound_file(
    root: Path, binding: Mapping[str, Any], field: str
) -> Path:
    value = str(binding[field])
    relative = Path(value)
    if relative.is_absolute() or not value or ".." in relative.parts:
        raise HumanPrecisionGateError("package_path_invalid")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HumanPrecisionGateError("package_path_outside_root") from exc
    return resolved


def _manifest_version(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("version")
    return "legacy_unversioned" if value is None else str(value)


def _opaque_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _missing_identity_digests(
    expected: set[str], observed: set[str]
) -> list[str]:
    return [_opaque_digest(value) for value in sorted(expected - observed)]
