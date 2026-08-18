"""Offline intake gate for locked blind human annotation submissions.

This module consumes the existing ``human_annotation_delivery_v1`` export
without defining another label format.  A small receipt binds that immutable
export to the already acknowledged assignment, while an append-only ledger
records intake state.  The operator-only mapping is used only after both
independent submissions are complete to create a blind disagreement queue.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.human_annotation_assignment_activation import (
    RECEIPT_PROTOCOL as ASSIGNMENT_RECEIPT_PROTOCOL,
    ROLES as ASSIGNMENT_ROLES,
    HumanAnnotationAssignmentError,
    canonical_json,
    load_protocol as load_assignment_protocol,
    read_object,
    sha256_bytes,
    sha256_file,
    verify_bundle,
    verify_event_chain as verify_assignment_event_chain,
    write_object,
)
from scholar_agent.evaluation.human_annotation_delivery import (
    CONTRACT as DELIVERY_CONTRACT,
    DeliveryError,
    DeliverySubmission,
    load_delivery_protocol,
    load_submission,
    submission_hash,
    verify_delivery,
)
from scholar_agent.evaluation.precision_annotation import LABELS


PROTOCOL = "human_annotation_submission_intake_v1"
RECEIPT_PROTOCOL = "human_annotation_submission_receipt_v1"
LEDGER_PROTOCOL = "human_annotation_submission_event_ledger_v1"
QUEUE_PROTOCOL = "human_annotation_adjudication_queue_v1"
QUEUE_MAPPING_PROTOCOL = "human_annotation_adjudication_queue_mapping_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "dcbede0928d6c15d0c9a68333e53837537fe1249"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ANNOTATOR_ROLES = ("annotator_a", "annotator_b")
ROLE_TO_SIDE = {"annotator_a": "A", "annotator_b": "B"}
STATES = (
    "awaiting_submissions",
    "one_submission_validated",
    "two_submissions_validated",
    "adjudication_queue_ready",
    "revoked",
    "invalid",
)
TRANSITIONS = {
    "awaiting_submissions": {
        "one_submission_validated",
        "revoked",
        "invalid",
    },
    "one_submission_validated": {
        "two_submissions_validated",
        "revoked",
        "invalid",
    },
    "two_submissions_validated": {
        "adjudication_queue_ready",
        "revoked",
        "invalid",
    },
    "adjudication_queue_ready": {"revoked", "invalid"},
}
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRINCIPAL_RE = re.compile(r"^prn_[0-9a-f]{16}$")
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "real_label_count": 0,
    "snapshot_write_count": 0,
}
FORMAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
FORBIDDEN_QUEUE_KEYS = frozenset(
    {
        "alias",
        "arm",
        "case_id",
        "global_opaque_id",
        "gold",
        "item_id",
        "operator_mapping",
        "package_role",
        "qrels",
        "rank",
        "score",
        "source",
        "strategy",
        "target_paper",
    }
)
MAX_JSON_BYTES = 4 * 1024 * 1024


class HumanAnnotationSubmissionError(RuntimeError):
    """Submission, receipt, ledger, or queue integrity is invalid."""


class HumanAnnotationSubmissionNotReady(HumanAnnotationSubmissionError):
    """Both complete real submissions are not yet available."""


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload[key] = ZERO_SHA256
    return sha256_bytes(canonical_json(payload))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise HumanAnnotationSubmissionError("unsafe_path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.as_posix() != value
        or path.name == ".env"
        or path.parts[0] == "third_party"
    ):
        raise HumanAnnotationSubmissionError("unsafe_path")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HumanAnnotationSubmissionError("json_input_unavailable") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise HumanAnnotationSubmissionError("json_input_too_large")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HumanAnnotationSubmissionError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise HumanAnnotationSubmissionError("json_input_invalid")
    return value


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = _read_json(path)
    if set(value) != {
        "bindings",
        "execution",
        "formal_validation_complete",
        "intake",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "state_machine",
        "synthetic_scenarios",
    }:
        raise HumanAnnotationSubmissionError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or _digest_without(value, "protocol_sha256") != value["protocol_sha256"]
    ):
        raise HumanAnnotationSubmissionError("protocol_binding_invalid")
    binding_names = {
        "adjudication",
        "assignment",
        "clearance",
        "delivery",
        "preregistration_seal",
        "qualification",
        "quarantine",
        "separation_of_duties",
    }
    if not isinstance(value["bindings"], dict) or set(value["bindings"]) != binding_names:
        raise HumanAnnotationSubmissionError("protocol_binding_inventory_invalid")
    for name, spec in value["bindings"].items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise HumanAnnotationSubmissionError("protocol_binding_schema_invalid")
        target = repository_root / _safe_relative(spec["path"])
        if not target.is_file() or sha256_file(target) != spec["sha256"]:
            raise HumanAnnotationSubmissionError(f"protocol_binding_drift:{name}")
    if value["state_machine"] != {
        "states": list(STATES),
        "transitions": {
            key: sorted(children) for key, children in TRANSITIONS.items()
        },
    }:
        raise HumanAnnotationSubmissionError("state_machine_drift")
    if value["intake"] != {
        "accepted_export_contract": DELIVERY_CONTRACT,
        "adjudication_visible_fields": [
            "disagreement_alias",
            "query",
            "title",
            "abstract",
            "year",
            "annotation_a",
            "annotation_b",
            "rubric",
        ],
        "all_acknowledgements_required": True,
        "complete_item_count_per_annotator": 471,
        "labels_before_dual_validation": "forbidden",
        "operator_mapping": "operator_only",
        "queue_population": "all_and_only_disagreements",
        "semantic_drift": "invalidate_without_label_migration",
        "statistics_before_adjudication": "forbidden",
    }:
        raise HumanAnnotationSubmissionError("intake_contract_drift")
    scenarios = [
        "valid_dual_submission",
        "only_a_submission",
        "partial_coverage",
        "duplicate_alias",
        "unknown_alias",
        "annotator_package_swap",
        "post_lock_tamper",
        "illegal_label",
        "coordinator_submission",
        "old_commit",
        "revoked_assignment",
        "disagreement_omission",
        "forged_adjudication_queue",
        "valid_reissue",
    ]
    if value["synthetic_scenarios"] != scenarios:
        raise HumanAnnotationSubmissionError("synthetic_scenario_contract_drift")
    return value


def _load_delivery(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    delivery_protocol_path = repository_root / protocol["bindings"]["delivery"]["path"]
    delivery_protocol = load_delivery_protocol(delivery_protocol_path, repository_root)
    package_root = repository_root / "benchmark/human_annotation_delivery_v1_release"
    verified = verify_delivery(delivery_protocol, package_root)
    if verified["item_count_per_annotator"] != 471:
        raise HumanAnnotationSubmissionError("delivery_population_invalid")
    return package_root, delivery_protocol, verified


def _assignment_receipt(
    path: Path,
    *,
    role: str,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_object(path)
    expected_keys = {
        "acknowledged",
        "assignment_challenge",
        "assignment_protocol_sha256",
        "bundle_sha256",
        "principal_id",
        "receipt_protocol",
        "receipt_sha256",
        "role",
        "schema_version",
        "source_commit",
        "state",
        "submitted_by_role",
    }
    if (
        set(value) != expected_keys
        or value["receipt_protocol"] != ASSIGNMENT_RECEIPT_PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["assignment_protocol_sha256"] != protocol["protocol_sha256"]
        or value["role"] != role
        or value["acknowledged"] is not True
        or value["state"] != "acknowledged"
        or value["submitted_by_role"] != "principal_self"
        or _digest_without(value, "receipt_sha256") != value["receipt_sha256"]
        or value["principal_id"] != manifest["principal_id"]
        or value["assignment_challenge"] != manifest["assignment_challenge"]
        or value["bundle_sha256"] != manifest["bundle_sha256"]
    ):
        raise HumanAnnotationSubmissionError("assignment_receipt_invalid")
    return value


def validate_assignment_context(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    bundle_paths: Mapping[str, Path],
    assignment_receipt_paths: Mapping[str, Path],
    assignment_ledger_path: Path,
) -> dict[str, Any]:
    if (
        set(bundle_paths) != set(ASSIGNMENT_ROLES)
        or set(assignment_receipt_paths) != set(ASSIGNMENT_ROLES)
    ):
        raise HumanAnnotationSubmissionError("assignment_population_invalid")
    assignment_protocol = load_assignment_protocol(
        repository_root / protocol["bindings"]["assignment"]["path"],
        repository_root=repository_root,
    )
    ledger = read_object(assignment_ledger_path)
    states = verify_assignment_event_chain(ledger)
    if set(states) != set(ASSIGNMENT_ROLES) or any(
        state != "locked_for_submission" for state in states.values()
    ):
        raise HumanAnnotationSubmissionNotReady("assignment_acknowledgements_incomplete")
    manifests: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    ledger_receipts = {
        str(row["role"]): str(row["receipt_sha256"])
        for row in ledger["receipts"]
        if isinstance(row, dict)
        and set(row) == {"receipt_sha256", "role"}
        and row.get("role") in ASSIGNMENT_ROLES
    }
    if set(ledger_receipts) != set(ASSIGNMENT_ROLES):
        raise HumanAnnotationSubmissionError("assignment_receipt_ledger_invalid")
    for role in ASSIGNMENT_ROLES:
        manifest = verify_bundle(
            bundle_paths[role],
            assignment_protocol,
            repository_root=repository_root,
            expected_role=role,
        )
        manifest["bundle_sha256"] = sha256_file(bundle_paths[role])
        receipt = _assignment_receipt(
            assignment_receipt_paths[role],
            role=role,
            manifest=manifest,
            protocol=assignment_protocol,
        )
        if ledger_receipts[role] != receipt["receipt_sha256"]:
            raise HumanAnnotationSubmissionError("assignment_receipt_ledger_mismatch")
        matching_events = [
            event
            for event in ledger["events"]
            if event["role"] == role
            and event["state"] == "locked_for_submission"
        ]
        if len(matching_events) != 1 or any(
            matching_events[0][key] != expected
            for key, expected in {
                "assignment_challenge": manifest["assignment_challenge"],
                "bundle_sha256": manifest["bundle_sha256"],
                "principal_id": manifest["principal_id"],
                "qualification_sha256": manifest["qualification_sha256"],
            }.items()
        ):
            raise HumanAnnotationSubmissionError("assignment_event_binding_mismatch")
        manifests[role] = manifest
        receipts[role] = receipt
    principals = [manifests[role]["principal_id"] for role in ASSIGNMENT_ROLES]
    if len(set(principals)) != len(principals):
        raise HumanAnnotationSubmissionError("assignment_principal_conflict")
    return {
        "assignment_protocol": assignment_protocol,
        "ledger_sha256": sha256_file(assignment_ledger_path),
        "manifests": manifests,
        "receipts": receipts,
    }


def build_submission_receipt(
    submission_path: Path,
    *,
    role: str,
    assignment_context: Mapping[str, Any],
    protocol: Mapping[str, Any],
    repository_root: Path,
    output: Path,
    submitted_by_role: str = "principal_self",
) -> None:
    if role not in ANNOTATOR_ROLES:
        raise HumanAnnotationSubmissionError("submission_role_invalid")
    package_root, delivery_protocol, _ = _load_delivery(repository_root, protocol)
    side = ROLE_TO_SIDE[role]
    submission = load_submission(
        submission_path,
        package_root=package_root,
        side=side,
        protocol=delivery_protocol,
    )
    manifest = assignment_context["manifests"][role]
    assignment_receipt = assignment_context["receipts"][role]
    value: dict[str, Any] = {
        "assignment_challenge": manifest["assignment_challenge"],
        "assignment_protocol_sha256": manifest["assignment_protocol_sha256"],
        "assignment_receipt_sha256": assignment_receipt["receipt_sha256"],
        "bundle_sha256": manifest["bundle_sha256"],
        "delivery_contract": DELIVERY_CONTRACT,
        "delivery_package_id": submission.package_id,
        "delivery_package_sha256": submission.package_sha256,
        "export_sha256": sha256_file(submission_path),
        "intake_protocol_sha256": protocol["protocol_sha256"],
        "item_count": len(submission.labels),
        "locked_labels_sha256": submission.labels_sha256,
        "principal_id": manifest["principal_id"],
        "receipt_protocol": RECEIPT_PROTOCOL,
        "receipt_sha256": ZERO_SHA256,
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "state": "locked_for_submission",
        "submitted_by_role": submitted_by_role,
        "synthetic_only": bool(manifest["synthetic_only"]),
    }
    value["receipt_sha256"] = _digest_without(value, "receipt_sha256")
    write_object(output, value)


def verify_locked_submission(
    submission_path: Path,
    submission_receipt_path: Path,
    *,
    role: str,
    assignment_context: Mapping[str, Any],
    protocol: Mapping[str, Any],
    repository_root: Path,
    allow_synthetic: bool = False,
) -> tuple[DeliverySubmission, dict[str, Any]]:
    if role not in ANNOTATOR_ROLES:
        raise HumanAnnotationSubmissionError("submission_role_invalid")
    package_root, delivery_protocol, _ = _load_delivery(repository_root, protocol)
    side = ROLE_TO_SIDE[role]
    try:
        submission = load_submission(
            submission_path,
            package_root=package_root,
            side=side,
            protocol=delivery_protocol,
        )
    except Exception as exc:
        if isinstance(exc, HumanAnnotationSubmissionError):
            raise
        raise HumanAnnotationSubmissionError(
            getattr(exc, "code", "locked_submission_invalid")
        ) from exc
    receipt = _read_json(submission_receipt_path)
    expected_keys = {
        "assignment_challenge",
        "assignment_protocol_sha256",
        "assignment_receipt_sha256",
        "bundle_sha256",
        "delivery_contract",
        "delivery_package_id",
        "delivery_package_sha256",
        "export_sha256",
        "intake_protocol_sha256",
        "item_count",
        "locked_labels_sha256",
        "principal_id",
        "receipt_protocol",
        "receipt_sha256",
        "role",
        "schema_version",
        "source_commit",
        "state",
        "submitted_by_role",
        "synthetic_only",
    }
    manifest = assignment_context["manifests"][role]
    assignment_receipt = assignment_context["receipts"][role]
    if (
        set(receipt) != expected_keys
        or receipt["receipt_protocol"] != RECEIPT_PROTOCOL
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["source_commit"] != SOURCE_COMMIT
        or receipt["intake_protocol_sha256"] != protocol["protocol_sha256"]
        or receipt["role"] != role
        or receipt["state"] != "locked_for_submission"
        or receipt["submitted_by_role"] != "principal_self"
        or receipt["delivery_contract"] != DELIVERY_CONTRACT
        or receipt["item_count"] != 471
        or _digest_without(receipt, "receipt_sha256")
        != receipt["receipt_sha256"]
        or receipt["assignment_challenge"] != manifest["assignment_challenge"]
        or receipt["assignment_protocol_sha256"]
        != manifest["assignment_protocol_sha256"]
        or receipt["assignment_receipt_sha256"]
        != assignment_receipt["receipt_sha256"]
        or receipt["bundle_sha256"] != manifest["bundle_sha256"]
        or receipt["principal_id"] != manifest["principal_id"]
        or receipt["delivery_package_id"] != submission.package_id
        or receipt["delivery_package_sha256"] != submission.package_sha256
        or receipt["export_sha256"] != sha256_file(submission_path)
        or receipt["locked_labels_sha256"] != submission.labels_sha256
        or receipt["synthetic_only"] is not bool(manifest["synthetic_only"])
        or (receipt["synthetic_only"] is True and not allow_synthetic)
    ):
        raise HumanAnnotationSubmissionError("submission_receipt_invalid")
    if len(submission.labels) != 471:
        raise HumanAnnotationSubmissionError("submission_coverage_invalid")
    return submission, receipt


def _new_ledger(protocol: Mapping[str, Any], assignment_context: Mapping[str, Any]) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "assignment_ledger_sha256": assignment_context["ledger_sha256"],
        "assignment_protocol_sha256": assignment_context["assignment_protocol"][
            "protocol_sha256"
        ],
        "events": [],
        "formal_validation_complete": False,
        "protocol": LEDGER_PROTOCOL,
        "queue": None,
        "source_commit": SOURCE_COMMIT,
        "submissions": [],
    }
    _append_event(
        ledger,
        state="awaiting_submissions",
        role=None,
        binding=None,
        queue_sha256=ZERO_SHA256,
    )
    return ledger


def _append_event(
    ledger: dict[str, Any],
    *,
    state: str,
    role: str | None,
    binding: Mapping[str, Any] | None,
    queue_sha256: str,
) -> None:
    previous = ledger["events"][-1]["event_sha256"] if ledger["events"] else ZERO_SHA256
    event: dict[str, Any] = {
        "assignment_receipt_sha256": (
            binding["assignment_receipt_sha256"] if binding else ZERO_SHA256
        ),
        "bundle_sha256": binding["bundle_sha256"] if binding else ZERO_SHA256,
        "event_sha256": ZERO_SHA256,
        "export_sha256": binding["export_sha256"] if binding else ZERO_SHA256,
        "locked_labels_sha256": (
            binding["locked_labels_sha256"] if binding else ZERO_SHA256
        ),
        "previous_sha256": previous,
        "principal_id": binding["principal_id"] if binding else None,
        "queue_sha256": queue_sha256,
        "role": role,
        "sequence": len(ledger["events"]) + 1,
        "source_commit": SOURCE_COMMIT,
        "state": state,
        "submission_receipt_sha256": (
            binding["receipt_sha256"] if binding else ZERO_SHA256
        ),
    }
    event["event_sha256"] = _digest_without(event, "event_sha256")
    ledger["events"].append(event)


def verify_event_chain(ledger: Mapping[str, Any]) -> str:
    if set(ledger) != {
        "assignment_ledger_sha256",
        "assignment_protocol_sha256",
        "events",
        "formal_validation_complete",
        "protocol",
        "queue",
        "source_commit",
        "submissions",
    } or (
        ledger["protocol"] != LEDGER_PROTOCOL
        or ledger["source_commit"] != SOURCE_COMMIT
        or ledger["formal_validation_complete"] is not False
        or not isinstance(ledger["events"], list)
        or not isinstance(ledger["submissions"], list)
        or not SHA256_RE.fullmatch(str(ledger["assignment_ledger_sha256"]))
        or not SHA256_RE.fullmatch(str(ledger["assignment_protocol_sha256"]))
    ):
        raise HumanAnnotationSubmissionError("submission_ledger_invalid")
    prior_state: str | None = None
    previous = ZERO_SHA256
    seen_roles: set[str] = set()
    submission_by_role: dict[str, Mapping[str, Any]] = {}
    for row in ledger["submissions"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "assignment_receipt_sha256",
                "bundle_sha256",
                "export_sha256",
                "locked_labels_sha256",
                "principal_id",
                "receipt_sha256",
                "role",
            }
            or row["role"] not in ANNOTATOR_ROLES
            or row["role"] in submission_by_role
        ):
            raise HumanAnnotationSubmissionError("submission_ledger_record_invalid")
        submission_by_role[str(row["role"])] = row
    for sequence, event in enumerate(ledger["events"], 1):
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "assignment_receipt_sha256",
                "bundle_sha256",
                "event_sha256",
                "export_sha256",
                "locked_labels_sha256",
                "previous_sha256",
                "principal_id",
                "queue_sha256",
                "role",
                "sequence",
                "source_commit",
                "state",
                "submission_receipt_sha256",
            }
            or event["sequence"] != sequence
            or event["previous_sha256"] != previous
            or event["source_commit"] != SOURCE_COMMIT
            or event["state"] not in STATES
            or _digest_without(event, "event_sha256") != event["event_sha256"]
        ):
            raise HumanAnnotationSubmissionError("submission_event_chain_invalid")
        state = str(event["state"])
        if prior_state is None:
            if state != "awaiting_submissions" or event["role"] is not None:
                raise HumanAnnotationSubmissionError("submission_state_invalid")
        elif state not in TRANSITIONS.get(prior_state, set()):
            raise HumanAnnotationSubmissionError("submission_transition_invalid")
        if state in {"one_submission_validated", "two_submissions_validated"}:
            role = event["role"]
            if role not in ANNOTATOR_ROLES or role in seen_roles:
                raise HumanAnnotationSubmissionError("submission_role_sequence_invalid")
            record = submission_by_role.get(str(role))
            if record is None or any(
                event[event_key] != record[record_key]
                for event_key, record_key in {
                    "assignment_receipt_sha256": "assignment_receipt_sha256",
                    "bundle_sha256": "bundle_sha256",
                    "export_sha256": "export_sha256",
                    "locked_labels_sha256": "locked_labels_sha256",
                    "principal_id": "principal_id",
                    "submission_receipt_sha256": "receipt_sha256",
                }.items()
            ):
                raise HumanAnnotationSubmissionError("submission_event_binding_invalid")
            seen_roles.add(str(role))
        elif state == "adjudication_queue_ready":
            if (
                event["role"] != "adjudicator"
                or not SHA256_RE.fullmatch(str(event["queue_sha256"]))
                or not isinstance(ledger["queue"], dict)
                or set(ledger["queue"])
                != {
                    "adjudicator_assignment_receipt_sha256",
                    "adjudicator_bundle_sha256",
                    "adjudicator_principal_id",
                    "item_count",
                    "mapping_sha256",
                    "queue_sha256",
                }
                or ledger["queue"].get("queue_sha256") != event["queue_sha256"]
                or ledger["queue"].get(
                    "adjudicator_assignment_receipt_sha256"
                )
                != event["assignment_receipt_sha256"]
                or ledger["queue"].get("adjudicator_bundle_sha256")
                != event["bundle_sha256"]
                or ledger["queue"].get("adjudicator_principal_id")
                != event["principal_id"]
                or not isinstance(ledger["queue"].get("item_count"), int)
                or ledger["queue"]["item_count"] < 0
                or not SHA256_RE.fullmatch(
                    str(ledger["queue"].get("mapping_sha256"))
                )
            ):
                raise HumanAnnotationSubmissionError("queue_event_binding_invalid")
        prior_state = state
        previous = str(event["event_sha256"])
    if prior_state is None:
        raise HumanAnnotationSubmissionError("submission_event_chain_empty")
    expected_count = {
        "awaiting_submissions": 0,
        "one_submission_validated": 1,
        "two_submissions_validated": 2,
        "adjudication_queue_ready": 2,
        "revoked": len(submission_by_role),
        "invalid": len(submission_by_role),
    }[prior_state]
    if len(submission_by_role) != expected_count:
        raise HumanAnnotationSubmissionError("submission_state_count_mismatch")
    return prior_state


def intake_submissions(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    ledger_path: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    if not submissions or not set(submissions).issubset(ANNOTATOR_ROLES):
        raise HumanAnnotationSubmissionError("submission_population_invalid")
    ledger = _read_json(ledger_path) if ledger_path.exists() else _new_ledger(
        protocol, assignment_context
    )
    state = verify_event_chain(ledger)
    if state in {"revoked", "invalid", "adjudication_queue_ready"}:
        raise HumanAnnotationSubmissionError("submission_ledger_not_writable")
    existing = {str(row["role"]) for row in ledger["submissions"]}
    for role in ANNOTATOR_ROLES:
        if role not in submissions:
            continue
        if role in existing:
            raise HumanAnnotationSubmissionError("duplicate_submission")
        submission, receipt = verify_locked_submission(
            submissions[role][0],
            submissions[role][1],
            role=role,
            assignment_context=assignment_context,
            protocol=protocol,
            repository_root=repository_root,
            allow_synthetic=allow_synthetic,
        )
        record = {
            "assignment_receipt_sha256": receipt["assignment_receipt_sha256"],
            "bundle_sha256": receipt["bundle_sha256"],
            "export_sha256": receipt["export_sha256"],
            "locked_labels_sha256": submission.labels_sha256,
            "principal_id": receipt["principal_id"],
            "receipt_sha256": receipt["receipt_sha256"],
            "role": role,
        }
        ledger["submissions"].append(record)
        next_state = (
            "one_submission_validated"
            if len(ledger["submissions"]) == 1
            else "two_submissions_validated"
        )
        _append_event(
            ledger,
            state=next_state,
            role=role,
            binding=receipt,
            queue_sha256=ZERO_SHA256,
        )
        existing.add(role)
    ledger["submissions"] = sorted(ledger["submissions"], key=lambda row: row["role"])
    final_state = verify_event_chain(ledger)
    write_object(ledger_path, ledger)
    ready = final_state == "two_submissions_validated"
    return {
        "adjudication_queue_allowed": ready,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY if ready else EXIT_NOT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": final_state,
        "statistics": None,
        "status": (
            "submission_chain_ready"
            if ready
            else "not_ready_missing_real_submissions"
        ),
        "validated_submission_count": len(ledger["submissions"]),
    }


def _expected_queue(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    allow_synthetic: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_root, _, _ = _load_delivery(repository_root, protocol)
    verified: dict[str, DeliverySubmission] = {}
    for role in ANNOTATOR_ROLES:
        if role not in submissions:
            raise HumanAnnotationSubmissionNotReady("two_complete_submissions_required")
        verified[role], _ = verify_locked_submission(
            submissions[role][0],
            submissions[role][1],
            role=role,
            assignment_context=assignment_context,
            protocol=protocol,
            repository_root=repository_root,
            allow_synthetic=allow_synthetic,
        )
    mapping = _read_json(package_root / "operator/mapping.json")
    if (
        set(mapping) != {"contract", "items", "mapping_sha256", "schema_version"}
        or mapping["contract"] != DELIVERY_CONTRACT
        or not isinstance(mapping["items"], list)
    ):
        raise HumanAnnotationSubmissionError("operator_mapping_invalid")
    lookups: dict[str, dict[str, Mapping[str, Any]]] = {}
    public: dict[str, dict[str, Mapping[str, Any]]] = {}
    label_rows: dict[str, dict[str, Any]] = {}
    for role, side in ROLE_TO_SIDE.items():
        lookups[side] = {
            str(row["alias"]): row
            for row in mapping["items"]
            if isinstance(row, dict) and row.get("side") == side
        }
        items = json.loads(
            (
                package_root / f"annotator-{side}/items.json"
            ).read_text(encoding="utf-8")
        )
        public[side] = {str(row["alias"]): row for row in items}
        label_rows[side] = {
            row.alias: {"label": row.label, "notes": row.notes}
            for row in verified[role].labels
        }
        if set(lookups[side]) != set(public[side]) or set(label_rows[side]) != set(
            public[side]
        ):
            raise HumanAnnotationSubmissionError("operator_mapping_coverage_invalid")
    identity_to_alias: dict[str, dict[str, str]] = {}
    for side in ("A", "B"):
        for alias, row in lookups[side].items():
            identity = f"{row['role']}:{row['item_id']}"
            identity_to_alias.setdefault(identity, {})[side] = alias
    if len(identity_to_alias) != 471 or any(
        set(sides) != {"A", "B"} for sides in identity_to_alias.values()
    ):
        raise HumanAnnotationSubmissionError("operator_mapping_pairing_invalid")
    rubric = _read_json(package_root / "annotator-A/rubric.json")
    rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for identity in sorted(identity_to_alias):
        aliases = identity_to_alias[identity]
        a_alias, b_alias = aliases["A"], aliases["B"]
        if (
            lookups["A"][a_alias]["content_sha256"]
            != lookups["B"][b_alias]["content_sha256"]
        ):
            raise HumanAnnotationSubmissionError("operator_mapping_content_mismatch")
        annotation_a = label_rows["A"][a_alias]
        annotation_b = label_rows["B"][b_alias]
        if annotation_a["label"] == annotation_b["label"]:
            continue
        disagreement_alias = "disagreement-" + hashlib.sha256(
            (
                protocol["protocol_sha256"]
                + "\0"
                + mapping["mapping_sha256"]
                + "\0"
                + identity
            ).encode("utf-8")
        ).hexdigest()[:24]
        public_a = public["A"][a_alias]
        rows.append(
            {
                "abstract": public_a.get("abstract"),
                "annotation_a": annotation_a,
                "annotation_b": annotation_b,
                "disagreement_alias": disagreement_alias,
                "query": public_a.get("query"),
                "title": public_a.get("title"),
                "year": public_a.get("year"),
            }
        )
        private_rows.append(
            {
                "disagreement_alias": disagreement_alias,
                "item_id": lookups["A"][a_alias]["item_id"],
                "package_role": lookups["A"][a_alias]["role"],
            }
        )
    rows.sort(key=lambda row: row["disagreement_alias"])
    private_rows.sort(key=lambda row: row["disagreement_alias"])
    adjudicator_receipt = assignment_context["receipts"]["adjudicator"]
    queue: dict[str, Any] = {
        "adjudicator_assignment_receipt_sha256": adjudicator_receipt[
            "receipt_sha256"
        ],
        "assignment_protocol_sha256": assignment_context["assignment_protocol"][
            "protocol_sha256"
        ],
        "formal_validation_complete": False,
        "intake_protocol_sha256": protocol["protocol_sha256"],
        "item_count": len(rows),
        "queue_protocol": QUEUE_PROTOCOL,
        "queue_sha256": ZERO_SHA256,
        "rows": rows,
        "rubric": rubric,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "statistics": None,
    }
    queue["queue_sha256"] = _digest_without(queue, "queue_sha256")
    private_mapping: dict[str, Any] = {
        "entries": private_rows,
        "intake_protocol_sha256": protocol["protocol_sha256"],
        "mapping_sha256": ZERO_SHA256,
        "protocol": QUEUE_MAPPING_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    private_mapping["mapping_sha256"] = _digest_without(
        private_mapping, "mapping_sha256"
    )
    return queue, private_mapping


def verify_adjudication_queue(
    queue_path: Path,
    operator_mapping_path: Path,
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    actual_queue = _read_json(queue_path)
    actual_mapping = _read_json(operator_mapping_path)
    expected_queue, expected_mapping = _expected_queue(
        repository_root,
        protocol,
        assignment_context=assignment_context,
        submissions=submissions,
        allow_synthetic=allow_synthetic,
    )
    if actual_queue != expected_queue:
        raise HumanAnnotationSubmissionError("adjudication_queue_population_invalid")
    if actual_mapping != expected_mapping:
        raise HumanAnnotationSubmissionError("adjudication_queue_mapping_invalid")
    if _walk_keys(actual_queue) & FORBIDDEN_QUEUE_KEYS:
        raise HumanAnnotationSubmissionError("adjudication_queue_blinding_violation")
    if any(
        row["annotation_a"]["label"] == row["annotation_b"]["label"]
        for row in actual_queue["rows"]
    ):
        raise HumanAnnotationSubmissionError("adjudication_without_disagreement")
    return actual_queue


def build_adjudication_queue(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    ledger_path: Path,
    queue_path: Path,
    operator_mapping_path: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    if verify_event_chain(ledger) != "two_submissions_validated":
        raise HumanAnnotationSubmissionNotReady("two_complete_submissions_required")
    if queue_path.exists() or operator_mapping_path.exists():
        raise HumanAnnotationSubmissionError("queue_target_not_empty")
    queue, private_mapping = _expected_queue(
        repository_root,
        protocol,
        assignment_context=assignment_context,
        submissions=submissions,
        allow_synthetic=allow_synthetic,
    )
    expected_bindings = {
        role: (
            sha256_file(paths[0]),
            _read_json(paths[1])["receipt_sha256"],
        )
        for role, paths in submissions.items()
    }
    ledger_bindings = {
        str(row["role"]): (row["export_sha256"], row["receipt_sha256"])
        for row in ledger["submissions"]
    }
    if expected_bindings != ledger_bindings:
        raise HumanAnnotationSubmissionError("queue_submission_binding_mismatch")
    write_object(queue_path, queue)
    write_object(operator_mapping_path, private_mapping)
    verify_adjudication_queue(
        queue_path,
        operator_mapping_path,
        repository_root=repository_root,
        protocol=protocol,
        assignment_context=assignment_context,
        submissions=submissions,
        allow_synthetic=allow_synthetic,
    )
    adjudicator_assignment = assignment_context["receipts"]["adjudicator"]
    ledger["queue"] = {
        "adjudicator_assignment_receipt_sha256": adjudicator_assignment[
            "receipt_sha256"
        ],
        "adjudicator_bundle_sha256": adjudicator_assignment["bundle_sha256"],
        "adjudicator_principal_id": adjudicator_assignment["principal_id"],
        "item_count": queue["item_count"],
        "mapping_sha256": private_mapping["mapping_sha256"],
        "queue_sha256": queue["queue_sha256"],
    }
    queue_binding = {
        "assignment_receipt_sha256": adjudicator_assignment["receipt_sha256"],
        "bundle_sha256": adjudicator_assignment["bundle_sha256"],
        "export_sha256": ZERO_SHA256,
        "locked_labels_sha256": ZERO_SHA256,
        "principal_id": adjudicator_assignment["principal_id"],
        "receipt_sha256": ZERO_SHA256,
    }
    _append_event(
        ledger,
        state="adjudication_queue_ready",
        role="adjudicator",
        binding=queue_binding,
        queue_sha256=queue["queue_sha256"],
    )
    verify_event_chain(ledger)
    write_object(ledger_path, ledger)
    return {
        "adjudication_queue_item_count": queue["item_count"],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": "adjudication_queue_ready",
        "statistics": None,
        "status": "submission_chain_ready",
    }


def _synthetic_assignment(
    root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    suffix: str = "",
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path], Path]:
    from scholar_agent.evaluation.human_annotation_assignment_activation import (
        build_acknowledgement,
        issue_assignments,
        verify_acknowledgements,
    )
    from scholar_agent.evaluation.human_annotator_qualification_intake import (
        build_contract,
        build_synthetic_submission,
        load_protocol as load_qualification_protocol,
    )

    assignment_protocol = load_assignment_protocol(
        repository_root / protocol["bindings"]["assignment"]["path"],
        repository_root=repository_root,
    )
    qualification_protocol = load_qualification_protocol(
        repository_root / protocol["bindings"]["qualification"]["path"],
        repository_root=repository_root,
    )
    qualifications = []
    for ordinal, role in enumerate(ASSIGNMENT_ROLES, 1):
        contract = build_contract(
            qualification_protocol,
            challenge=sha256_bytes(f"qualification:{suffix}:{role}".encode()),
            role=role,
        )
        path = root / f"qualification-{suffix}-{role}.json"
        build_synthetic_submission(
            contract,
            path,
            principal_id=f"prn_{ordinal:016x}",
            principal_commitment=sha256_bytes(
                f"principal:{suffix}:{role}".encode()
            ),
        )
        qualifications.append(_read_json(path))
    bundles_root = root / f"bundles-{suffix}"
    assignment_ledger = root / f"assignment-ledger-{suffix}.json"
    challenges = {
        role: sha256_bytes(f"assignment:{suffix}:{role}".encode())
        for role in ASSIGNMENT_ROLES
    }
    issue_assignments(
        repository_root,
        assignment_protocol,
        qualifications,
        challenges=challenges,
        output_root=bundles_root,
        ledger_path=assignment_ledger,
        allow_synthetic=True,
    )
    bundles = {
        role: bundles_root / f"{role}.zip" for role in ASSIGNMENT_ROLES
    }
    receipts: dict[str, Path] = {}
    for role in ASSIGNMENT_ROLES:
        path = root / f"assignment-receipt-{suffix}-{role}.json"
        build_acknowledgement(
            bundles[role],
            assignment_protocol,
            repository_root=repository_root,
            output=path,
        )
        receipts[role] = path
    verify_acknowledgements(
        bundles,
        list(receipts.values()),
        assignment_ledger,
        assignment_protocol,
        repository_root=repository_root,
    )
    context = validate_assignment_context(
        repository_root,
        protocol,
        bundle_paths=bundles,
        assignment_receipt_paths=receipts,
        assignment_ledger_path=assignment_ledger,
    )
    return context, bundles, receipts, assignment_ledger


def _synthetic_export(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    side: str,
    output: Path,
    offset: int,
) -> None:
    package_root, _, _ = _load_delivery(repository_root, protocol)
    package = _read_json(package_root / f"annotator-{side}/package.json")
    items = json.loads(
        (package_root / f"annotator-{side}/items.json").read_text(encoding="utf-8")
    )
    labels = list(LABELS)
    payload: dict[str, Any] = {
        "annotator_id": package["annotator_id"],
        "contract": DELIVERY_CONTRACT,
        "labels": [
            {
                "alias": row["alias"],
                "label": labels[(index + offset) % len(labels)],
                "notes": "",
            }
            for index, row in enumerate(items)
        ],
        "labels_sha256": ZERO_SHA256,
        "locked": True,
        "package_id": package["package_id"],
        "package_sha256": package["package_sha256"],
        "schema_version": SCHEMA_VERSION,
        "side": side,
    }
    payload["labels_sha256"] = submission_hash(payload)
    write_object(output, payload)


def _scenario_inputs(
    root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    suffix: str = "",
) -> tuple[
    dict[str, Any],
    dict[str, Path],
    dict[str, Path],
    Path,
    dict[str, tuple[Path, Path]],
]:
    context, bundles, assignment_receipts, assignment_ledger = (
        _synthetic_assignment(root, repository_root, protocol, suffix=suffix)
    )
    submissions: dict[str, tuple[Path, Path]] = {}
    for offset, role in enumerate(ANNOTATOR_ROLES):
        export = root / f"submission-{suffix}-{role}.json"
        receipt = root / f"submission-receipt-{suffix}-{role}.json"
        _synthetic_export(
            repository_root,
            protocol,
            side=ROLE_TO_SIDE[role],
            output=export,
            offset=offset,
        )
        build_submission_receipt(
            export,
            role=role,
            assignment_context=context,
            protocol=protocol,
            repository_root=repository_root,
            output=receipt,
        )
        submissions[role] = (export, receipt)
    return context, bundles, assignment_receipts, assignment_ledger, submissions


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="human-annotation-submission-matrix-"
    ) as temp_name:
        base = Path(temp_name)
        for scenario in protocol["synthetic_scenarios"]:
            root = base / scenario
            root.mkdir()
            expected = (
                "passed"
                if scenario in {"valid_dual_submission", "valid_reissue"}
                else "not_ready"
                if scenario == "only_a_submission"
                else "rejected"
            )
            observed = "passed"
            reason = None
            try:
                context, bundles, assignment_receipts, assignment_ledger, inputs = (
                    _scenario_inputs(root, repository_root, protocol)
                )
                ledger = root / "intake-ledger.json"
                if scenario == "only_a_submission":
                    result = intake_submissions(
                        repository_root,
                        protocol,
                        assignment_context=context,
                        submissions={"annotator_a": inputs["annotator_a"]},
                        ledger_path=ledger,
                        allow_synthetic=True,
                    )
                    observed = (
                        "not_ready"
                        if result["state"] == "one_submission_validated"
                        else "rejected"
                    )
                else:
                    if scenario in {
                        "partial_coverage",
                        "duplicate_alias",
                        "unknown_alias",
                        "post_lock_tamper",
                        "illegal_label",
                    }:
                        path, receipt_path = inputs["annotator_a"]
                        value = _read_json(path)
                        if scenario == "partial_coverage":
                            value["labels"] = value["labels"][:-1]
                        elif scenario == "duplicate_alias":
                            value["labels"][-1]["alias"] = value["labels"][0]["alias"]
                        elif scenario == "unknown_alias":
                            value["labels"][-1]["alias"] = "item-" + "f" * 24
                        elif scenario == "illegal_label":
                            value["labels"][0]["label"] = "not-a-rubric-label"
                        elif scenario == "post_lock_tamper":
                            value["labels"][0]["notes"] = "changed-after-lock"
                            write_object(path, value)
                        if scenario != "post_lock_tamper":
                            value["labels_sha256"] = submission_hash(value)
                            write_object(path, value)
                            build_submission_receipt(
                                path,
                                role="annotator_a",
                                assignment_context=context,
                                protocol=protocol,
                                repository_root=repository_root,
                                output=receipt_path,
                            )
                    elif scenario == "annotator_package_swap":
                        inputs["annotator_a"], inputs["annotator_b"] = (
                            inputs["annotator_b"],
                            inputs["annotator_a"],
                        )
                    elif scenario == "coordinator_submission":
                        receipt = _read_json(inputs["annotator_a"][1])
                        receipt["submitted_by_role"] = "human_package_coordinator"
                        receipt["receipt_sha256"] = _digest_without(
                            receipt, "receipt_sha256"
                        )
                        write_object(inputs["annotator_a"][1], receipt)
                    elif scenario == "old_commit":
                        receipt = _read_json(inputs["annotator_a"][1])
                        receipt["source_commit"] = "0" * 40
                        receipt["receipt_sha256"] = _digest_without(
                            receipt, "receipt_sha256"
                        )
                        write_object(inputs["annotator_a"][1], receipt)
                    elif scenario == "revoked_assignment":
                        from scholar_agent.evaluation.human_annotation_assignment_activation import (
                            _append_state as append_assignment_state,
                        )

                        assignment_state = read_object(assignment_ledger)
                        manifest = copy.deepcopy(
                            context["manifests"]["annotator_a"]
                        )
                        append_assignment_state(
                            assignment_state, manifest, "revoked"
                        )
                        write_object(assignment_ledger, assignment_state)
                        context = validate_assignment_context(
                            repository_root,
                            protocol,
                            bundle_paths=bundles,
                            assignment_receipt_paths=assignment_receipts,
                            assignment_ledger_path=assignment_ledger,
                        )
                    intake_submissions(
                        repository_root,
                        protocol,
                        assignment_context=context,
                        submissions=inputs,
                        ledger_path=ledger,
                        allow_synthetic=True,
                    )
                    queue = root / "queue.json"
                    mapping = root / "queue-mapping.json"
                    build_adjudication_queue(
                        repository_root,
                        protocol,
                        assignment_context=context,
                        submissions=inputs,
                        ledger_path=ledger,
                        queue_path=queue,
                        operator_mapping_path=mapping,
                        allow_synthetic=True,
                    )
                    if scenario in {
                        "disagreement_omission",
                        "forged_adjudication_queue",
                    }:
                        value = _read_json(queue)
                        if scenario == "disagreement_omission":
                            value["rows"] = value["rows"][:-1]
                            value["item_count"] -= 1
                        else:
                            forged = copy.deepcopy(value["rows"][0])
                            forged["disagreement_alias"] = (
                                "disagreement-" + "f" * 24
                            )
                            value["rows"].append(forged)
                            value["item_count"] += 1
                        value["queue_sha256"] = _digest_without(
                            value, "queue_sha256"
                        )
                        write_object(queue, value)
                        verify_adjudication_queue(
                            queue,
                            mapping,
                            repository_root=repository_root,
                            protocol=protocol,
                            assignment_context=context,
                            submissions=inputs,
                            allow_synthetic=True,
                        )
                    elif scenario == "valid_reissue":
                        new_root = root / "reissue"
                        new_root.mkdir()
                        (
                            new_context,
                            _,
                            _,
                            _,
                            new_inputs,
                        ) = _scenario_inputs(
                            new_root,
                            repository_root,
                            protocol,
                            suffix="reissue",
                        )
                        try:
                            verify_locked_submission(
                                inputs["annotator_a"][0],
                                inputs["annotator_a"][1],
                                role="annotator_a",
                                assignment_context=new_context,
                                protocol=protocol,
                                repository_root=repository_root,
                                allow_synthetic=True,
                            )
                        except HumanAnnotationSubmissionError:
                            pass
                        else:
                            raise HumanAnnotationSubmissionError(
                                "reissue_inherited_old_labels"
                            )
                        reissue_ledger = new_root / "intake-ledger.json"
                        intake_submissions(
                            repository_root,
                            protocol,
                            assignment_context=new_context,
                            submissions=new_inputs,
                            ledger_path=reissue_ledger,
                            allow_synthetic=True,
                        )
            except (
                HumanAnnotationSubmissionError,
                HumanAnnotationAssignmentError,
                DeliveryError,
            ) as exc:
                observed = "rejected"
                reason = str(exc)
            rows.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "reason": reason,
                    "scenario": scenario,
                }
            )
    if any(row["expected"] != row["observed"] for row in rows):
        raise HumanAnnotationSubmissionError("synthetic_matrix_mismatch")
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "passed_count": len(rows),
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "scenario_count": len(rows),
        "scenarios": rows,
        "schema_version": SCHEMA_VERSION,
        "statistics": None,
        "status": "submission_chain_ready",
        "synthetic_artifacts_persisted": False,
    }


def audit_readiness(_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blocked_reasons": [
            "annotator_a_real_qualification_missing",
            "annotator_b_real_qualification_missing",
            "adjudicator_real_qualification_missing",
            "annotator_a_assignment_acknowledgement_missing",
            "annotator_b_assignment_acknowledgement_missing",
            "adjudicator_assignment_acknowledgement_missing",
            "annotator_a_locked_submission_missing",
            "annotator_b_locked_submission_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": "not_ready_missing_real_submissions",
        "statistics": None,
        "status": "not_ready_missing_real_submissions",
    }
