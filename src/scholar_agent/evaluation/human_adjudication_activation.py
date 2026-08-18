"""Issue blind adjudication tasks and unlock the existing human scorer.

The module deliberately adds no label or scoring semantics.  It binds the
disagreement-only queue produced by ``human_annotation_submission_intake_v1``
to the assigned adjudicator, validates a locked result, and delegates the
complete 439+32 item chain to ``human_precision_adjudication_v1``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.human_annotation_delivery import (
    ingest as ingest_delivery,
    load_delivery_protocol,
)
from scholar_agent.evaluation.human_annotation_submission_intake import (
    HumanAnnotationSubmissionError,
    _append_event as _append_submission_event,
    _scenario_inputs,
    build_submission_receipt,
    build_adjudication_queue,
    intake_submissions,
    load_protocol as load_submission_protocol,
    validate_assignment_context,
    verify_adjudication_queue,
    verify_event_chain as verify_submission_event_chain,
)
from scholar_agent.evaluation.human_annotation_delivery import submission_hash
from scholar_agent.evaluation.human_precision_adjudication import (
    AdjudicationRow,
    AdjudicationSubmission,
    IndependentSubmission,
    LabelRow,
    PriorLabelRow,
    PriorResolvedSubmission,
    load_protocol as load_precision_protocol,
    run_human_precision_gate,
    validate_package,
)
from scholar_agent.evaluation.precision_annotation import LABELS


PROTOCOL = "human_adjudication_activation_v1"
PACKAGE_PROTOCOL = "human_adjudication_package_v1"
ACK_PROTOCOL = "human_adjudication_acknowledgement_v1"
RESULT_PROTOCOL = "human_adjudication_result_v1"
LEDGER_PROTOCOL = "human_adjudication_event_ledger_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "e8add06c729156223466d51e8f718cdfb59e9035"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRINCIPAL_RE = re.compile(r"^prn_[0-9a-f]{16}$")
STATES = (
    "queue_ready",
    "issued",
    "acknowledged",
    "adjudication_submitted",
    "validated",
    "statistics_eligible",
    "revoked",
    "invalid",
)
TRANSITIONS = {
    "queue_ready": {"issued", "revoked", "invalid"},
    "issued": {"acknowledged", "revoked", "invalid"},
    "acknowledged": {"adjudication_submitted", "revoked", "invalid"},
    "adjudication_submitted": {"validated", "revoked", "invalid"},
    "validated": {"statistics_eligible", "revoked", "invalid"},
    "statistics_eligible": {"revoked", "invalid"},
}
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "official_scorer_call_count": 0,
    "quality_metric_count": 0,
    "real_label_count": 0,
    "snapshot_write_count": 0,
}
FORMAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_RATIONALE = 1000
PACKAGE_FIELDS = {
    "acknowledgement_required",
    "activation_protocol_sha256",
    "adjudicator_assignment_receipt_sha256",
    "adjudicator_principal_id",
    "challenge",
    "formal_validation_complete",
    "item_count",
    "package_protocol",
    "package_sha256",
    "queue_sha256",
    "rows",
    "rubric",
    "schema_version",
    "source_commit",
    "submission_hashes",
    "synthetic_only",
}
FORBIDDEN_PACKAGE_KEYS = frozenset(
    {
        "arm",
        "case_id",
        "global_opaque_id",
        "gold",
        "item_id",
        "mapping",
        "package_role",
        "qrels",
        "rank",
        "score",
        "source",
        "strategy",
        "target_paper",
    }
)


class HumanAdjudicationActivationError(RuntimeError):
    """An adjudication package, result, or event chain is invalid."""


class HumanAdjudicationActivationNotReady(HumanAdjudicationActivationError):
    """Complete fresh real evidence is not yet available."""


def canonical_json(value: Any) -> bytes:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload[key] = ZERO_SHA256
    return sha256_bytes(canonical_json(payload))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HumanAdjudicationActivationError("artifact_unavailable") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise HumanAdjudicationActivationError("artifact_size_exceeded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HumanAdjudicationActivationError("artifact_json_invalid") from exc
    if not isinstance(value, dict):
        raise HumanAdjudicationActivationError("artifact_root_invalid")
    return value


def write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(dict(value)))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise HumanAdjudicationActivationError("unsafe_binding_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise HumanAdjudicationActivationError("unsafe_binding_path")
    return str(path)


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "adjudication",
        "bindings",
        "execution",
        "formal_validation_complete",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "state_machine",
        "synthetic_scenarios",
    }:
        raise HumanAdjudicationActivationError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or value["state_machine"].get("states") != list(STATES)
        or value["state_machine"].get("transitions")
        != {key: sorted(rows) for key, rows in TRANSITIONS.items()}
        or _digest_without(value, "protocol_sha256") != value["protocol_sha256"]
    ):
        raise HumanAdjudicationActivationError("protocol_binding_invalid")
    expected_bindings = {
        "assignment",
        "clearance",
        "human_precision_adjudication",
        "preregistration",
        "qualification",
        "quarantine",
        "separation_of_duties",
        "submission_intake",
    }
    if set(value["bindings"]) != expected_bindings:
        raise HumanAdjudicationActivationError("protocol_binding_invalid")
    for binding in value["bindings"].values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise HumanAdjudicationActivationError("protocol_binding_invalid")
        target = repository_root / _safe_relative(binding["path"])
        if not target.is_file() or sha256_file(target) != binding["sha256"]:
            raise HumanAdjudicationActivationError("protocol_dependency_drift")
    adjudication = value["adjudication"]
    if (
        adjudication.get("complete_delivery_item_count") != 471
        or adjudication.get("complete_current_item_count") != 439
        or adjudication.get("complete_prior_chain_item_count") != 32
        or adjudication.get("dispute_population")
        != "all_and_only_submission_intake_disagreements"
        or adjudication.get("statistics_unlock")
        != "existing_change_only_cluster_aware_scorer_only"
        or adjudication.get("absolute_precision_at_20")
        != "unsupported_from_change_only_package"
    ):
        raise HumanAdjudicationActivationError("protocol_adjudication_invalid")
    return value


def _validate_queue(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    submission_ledger_path: Path,
    queue_path: Path,
    operator_mapping_path: Path,
    allow_synthetic: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    submission_protocol = load_submission_protocol(
        repository_root / protocol["bindings"]["submission_intake"]["path"],
        repository_root=repository_root,
    )
    ledger = read_object(submission_ledger_path)
    if verify_submission_event_chain(ledger) != "adjudication_queue_ready":
        raise HumanAdjudicationActivationNotReady("adjudication_queue_not_ready")
    queue = verify_adjudication_queue(
        queue_path,
        operator_mapping_path,
        repository_root=repository_root,
        protocol=submission_protocol,
        assignment_context=assignment_context,
        submissions=submissions,
        allow_synthetic=allow_synthetic,
    )
    if ledger["queue"]["queue_sha256"] != queue["queue_sha256"]:
        raise HumanAdjudicationActivationError("queue_ledger_binding_mismatch")
    mapping = read_object(operator_mapping_path)
    return queue, mapping


def _append_event(
    ledger: dict[str, Any],
    *,
    state: str,
    package_sha256: str,
    result_sha256: str = ZERO_SHA256,
) -> None:
    previous = ledger["events"][-1]["event_sha256"] if ledger["events"] else ZERO_SHA256
    event = {
        "event_sha256": ZERO_SHA256,
        "package_sha256": package_sha256,
        "previous_sha256": previous,
        "queue_sha256": ledger["queue_sha256"],
        "result_sha256": result_sha256,
        "sequence": len(ledger["events"]) + 1,
        "source_commit": SOURCE_COMMIT,
        "state": state,
    }
    event["event_sha256"] = _digest_without(event, "event_sha256")
    ledger["events"].append(event)


def verify_event_chain(ledger: Mapping[str, Any]) -> str:
    if set(ledger) != {
        "activation_protocol_sha256",
        "adjudicator_principal_id",
        "events",
        "formal_validation_complete",
        "ledger_protocol",
        "package_sha256",
        "queue_sha256",
        "source_commit",
        "synthetic_only",
    } or (
        ledger["ledger_protocol"] != LEDGER_PROTOCOL
        or ledger["source_commit"] != SOURCE_COMMIT
        or ledger["formal_validation_complete"] is not False
        or not isinstance(ledger["events"], list)
        or not PRINCIPAL_RE.fullmatch(str(ledger["adjudicator_principal_id"]))
    ):
        raise HumanAdjudicationActivationError("adjudication_ledger_invalid")
    previous = ZERO_SHA256
    prior: str | None = None
    for sequence, event in enumerate(ledger["events"], 1):
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "event_sha256",
                "package_sha256",
                "previous_sha256",
                "queue_sha256",
                "result_sha256",
                "sequence",
                "source_commit",
                "state",
            }
            or event["sequence"] != sequence
            or event["previous_sha256"] != previous
            or event["queue_sha256"] != ledger["queue_sha256"]
            or event["source_commit"] != SOURCE_COMMIT
            or event["state"] not in STATES
            or _digest_without(event, "event_sha256") != event["event_sha256"]
        ):
            raise HumanAdjudicationActivationError("adjudication_event_chain_invalid")
        state = str(event["state"])
        if prior is None:
            if state != "queue_ready":
                raise HumanAdjudicationActivationError("adjudication_state_invalid")
        elif state not in TRANSITIONS.get(prior, set()):
            raise HumanAdjudicationActivationError("adjudication_transition_invalid")
        previous, prior = str(event["event_sha256"]), state
    if prior is None:
        raise HumanAdjudicationActivationError("adjudication_event_chain_empty")
    return prior


def issue_package(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    assignment_context: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
    submission_ledger_path: Path,
    queue_path: Path,
    operator_mapping_path: Path,
    challenge: str,
    package_path: Path,
    ledger_path: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(challenge):
        raise HumanAdjudicationActivationError("challenge_invalid")
    if package_path.exists() or ledger_path.exists():
        raise HumanAdjudicationActivationError("issuance_target_not_empty")
    queue, _ = _validate_queue(
        repository_root,
        protocol,
        assignment_context=assignment_context,
        submissions=submissions,
        submission_ledger_path=submission_ledger_path,
        queue_path=queue_path,
        operator_mapping_path=operator_mapping_path,
        allow_synthetic=allow_synthetic,
    )
    receipt = assignment_context["receipts"]["adjudicator"]
    synthetic_only = bool(
        assignment_context["manifests"]["adjudicator"]["synthetic_only"]
    )
    if synthetic_only and not allow_synthetic:
        raise HumanAdjudicationActivationNotReady("real_adjudicator_required")
    rows = [
        {
            "abstract": row["abstract"],
            "annotation_a": row["annotation_a"],
            "annotation_b": row["annotation_b"],
            "disagreement_alias": row["disagreement_alias"],
            "query": row["query"],
            "title": row["title"],
            "year": row["year"],
        }
        for row in queue["rows"]
    ]
    package: dict[str, Any] = {
        "acknowledgement_required": True,
        "activation_protocol_sha256": protocol["protocol_sha256"],
        "adjudicator_assignment_receipt_sha256": receipt["receipt_sha256"],
        "adjudicator_principal_id": receipt["principal_id"],
        "challenge": challenge,
        "formal_validation_complete": False,
        "item_count": len(rows),
        "package_protocol": PACKAGE_PROTOCOL,
        "package_sha256": ZERO_SHA256,
        "queue_sha256": queue["queue_sha256"],
        "rows": rows,
        "rubric": queue["rubric"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "submission_hashes": {
            role: sha256_file(paths[0]) for role, paths in sorted(submissions.items())
        },
        "synthetic_only": synthetic_only,
    }
    if _walk_keys(package) & FORBIDDEN_PACKAGE_KEYS:
        raise HumanAdjudicationActivationError("adjudication_package_blinding_violation")
    package["package_sha256"] = _digest_without(package, "package_sha256")
    ledger: dict[str, Any] = {
        "activation_protocol_sha256": protocol["protocol_sha256"],
        "adjudicator_principal_id": receipt["principal_id"],
        "events": [],
        "formal_validation_complete": False,
        "ledger_protocol": LEDGER_PROTOCOL,
        "package_sha256": package["package_sha256"],
        "queue_sha256": queue["queue_sha256"],
        "source_commit": SOURCE_COMMIT,
        "synthetic_only": synthetic_only,
    }
    _append_event(ledger, state="queue_ready", package_sha256=ZERO_SHA256)
    _append_event(
        ledger, state="issued", package_sha256=package["package_sha256"]
    )
    verify_event_chain(ledger)
    write_object(package_path, package)
    write_object(ledger_path, ledger)
    return _status("issued", package["item_count"])


def verify_package(
    package_path: Path,
    protocol: Mapping[str, Any],
    *,
    expected_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    package = read_object(package_path)
    if set(package) != PACKAGE_FIELDS or (
        package["package_protocol"] != PACKAGE_PROTOCOL
        or package["schema_version"] != SCHEMA_VERSION
        or package["source_commit"] != SOURCE_COMMIT
        or package["activation_protocol_sha256"] != protocol["protocol_sha256"]
        or package["formal_validation_complete"] is not False
        or package["acknowledgement_required"] is not True
        or not isinstance(package["rows"], list)
        or package["item_count"] != len(package["rows"])
        or not PRINCIPAL_RE.fullmatch(str(package["adjudicator_principal_id"]))
        or _digest_without(package, "package_sha256") != package["package_sha256"]
    ):
        raise HumanAdjudicationActivationError("adjudication_package_invalid")
    aliases = [row.get("disagreement_alias") for row in package["rows"]]
    if (
        any(not isinstance(row, dict) for row in package["rows"])
        or any(
            set(row)
            != {
                "abstract",
                "annotation_a",
                "annotation_b",
                "disagreement_alias",
                "query",
                "title",
                "year",
            }
            for row in package["rows"]
        )
        or len(set(aliases)) != len(aliases)
        or aliases != sorted(aliases)
        or _walk_keys(package) & FORBIDDEN_PACKAGE_KEYS
    ):
        raise HumanAdjudicationActivationError("adjudication_package_population_invalid")
    if expected_queue is not None and (
        package["queue_sha256"] != expected_queue["queue_sha256"]
        or package["rows"] != expected_queue["rows"]
        or package["rubric"] != expected_queue["rubric"]
    ):
        raise HumanAdjudicationActivationError("adjudication_package_queue_mismatch")
    return package


def build_acknowledgement(package_path: Path, output: Path) -> None:
    package = read_object(package_path)
    value: dict[str, Any] = {
        "acknowledgement_protocol": ACK_PROTOCOL,
        "acknowledgement_sha256": ZERO_SHA256,
        "adjudicator_principal_id": package["adjudicator_principal_id"],
        "challenge": package["challenge"],
        "package_sha256": package["package_sha256"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "state": "acknowledged",
        "synthetic_only": package["synthetic_only"],
    }
    value["acknowledgement_sha256"] = _digest_without(
        value, "acknowledgement_sha256"
    )
    write_object(output, value)


def build_synthetic_result(
    package_path: Path,
    output: Path,
    *,
    choose: str = "annotation_a",
) -> None:
    package = read_object(package_path)
    decisions = [
        {
            "disagreement_alias": row["disagreement_alias"],
            "final_label": row[choose]["label"],
            "rationale": "synthetic rubric decision",
        }
        for row in package["rows"]
    ]
    value: dict[str, Any] = {
        "adjudicator_principal_id": package["adjudicator_principal_id"],
        "challenge": package["challenge"],
        "decisions": decisions,
        "decisions_sha256": sha256_bytes(canonical_json(decisions)),
        "locked": True,
        "package_sha256": package["package_sha256"],
        "result_protocol": RESULT_PROTOCOL,
        "result_sha256": ZERO_SHA256,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "synthetic_only": package["synthetic_only"],
    }
    value["result_sha256"] = _digest_without(value, "result_sha256")
    write_object(output, value)


def _verify_acknowledgement(
    acknowledgement_path: Path, package: Mapping[str, Any]
) -> dict[str, Any]:
    value = read_object(acknowledgement_path)
    if set(value) != {
        "acknowledgement_protocol",
        "acknowledgement_sha256",
        "adjudicator_principal_id",
        "challenge",
        "package_sha256",
        "schema_version",
        "source_commit",
        "state",
        "synthetic_only",
    } or (
        value["acknowledgement_protocol"] != ACK_PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["state"] != "acknowledged"
        or value["package_sha256"] != package["package_sha256"]
        or value["challenge"] != package["challenge"]
        or value["adjudicator_principal_id"]
        != package["adjudicator_principal_id"]
        or value["synthetic_only"] is not package["synthetic_only"]
        or _digest_without(value, "acknowledgement_sha256")
        != value["acknowledgement_sha256"]
    ):
        raise HumanAdjudicationActivationError("acknowledgement_invalid")
    return value


def _verify_submission_ledger_active(
    submission_ledger_path: Path,
    package: Mapping[str, Any],
) -> None:
    ledger = read_object(submission_ledger_path)
    try:
        state = verify_submission_event_chain(ledger)
    except HumanAnnotationSubmissionError as exc:
        raise HumanAdjudicationActivationError(
            "upstream_submission_ledger_invalid"
        ) from exc
    if state != "adjudication_queue_ready":
        raise HumanAdjudicationActivationError("upstream_submission_revoked")
    queue = ledger.get("queue")
    if (
        not isinstance(queue, dict)
        or queue.get("queue_sha256") != package["queue_sha256"]
    ):
        raise HumanAdjudicationActivationError(
            "upstream_submission_queue_mismatch"
        )


def _validate_locked_result(
    result_path: Path,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    result = read_object(result_path)
    if set(result) != {
        "adjudicator_principal_id",
        "challenge",
        "decisions",
        "decisions_sha256",
        "locked",
        "package_sha256",
        "result_protocol",
        "result_sha256",
        "schema_version",
        "source_commit",
        "synthetic_only",
    } or (
        result["result_protocol"] != RESULT_PROTOCOL
        or result["schema_version"] != SCHEMA_VERSION
        or result["source_commit"] != SOURCE_COMMIT
        or result["locked"] is not True
        or result["package_sha256"] != package["package_sha256"]
        or result["challenge"] != package["challenge"]
        or result["adjudicator_principal_id"]
        != package["adjudicator_principal_id"]
        or result["synthetic_only"] is not package["synthetic_only"]
        or not isinstance(result["decisions"], list)
        or sha256_bytes(canonical_json(result["decisions"]))
        != result["decisions_sha256"]
        or _digest_without(result, "result_sha256") != result["result_sha256"]
    ):
        raise HumanAdjudicationActivationError("adjudication_result_invalid")
    expected = [row["disagreement_alias"] for row in package["rows"]]
    actual: list[str] = []
    for row in result["decisions"]:
        if not isinstance(row, dict) or set(row) != {
            "disagreement_alias",
            "final_label",
            "rationale",
        }:
            raise HumanAdjudicationActivationError("adjudication_decision_invalid")
        alias = row["disagreement_alias"]
        rationale = row["rationale"]
        if (
            not isinstance(alias, str)
            or row["final_label"] not in LABELS
            or not isinstance(rationale, str)
            or not rationale.strip()
            or len(rationale) > MAX_RATIONALE
        ):
            raise HumanAdjudicationActivationError("adjudication_decision_invalid")
        actual.append(alias)
    if len(set(actual)) != len(actual):
        raise HumanAdjudicationActivationError("duplicate_adjudication_decision")
    if actual != expected:
        raise HumanAdjudicationActivationError("adjudication_coverage_invalid")
    return result


def verify_result(
    package_path: Path,
    acknowledgement_path: Path,
    result_path: Path,
    ledger_path: Path,
    submission_ledger_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    package = verify_package(package_path, protocol)
    _verify_submission_ledger_active(submission_ledger_path, package)
    _verify_acknowledgement(acknowledgement_path, package)
    result = _validate_locked_result(result_path, package)
    expected = [row["disagreement_alias"] for row in package["rows"]]
    ledger = read_object(ledger_path)
    if verify_event_chain(ledger) != "issued":
        raise HumanAdjudicationActivationError("duplicate_or_invalid_adjudication")
    if (
        ledger["package_sha256"] != package["package_sha256"]
        or ledger["adjudicator_principal_id"]
        != package["adjudicator_principal_id"]
    ):
        raise HumanAdjudicationActivationError("adjudication_ledger_binding_mismatch")
    _append_event(
        ledger, state="acknowledged", package_sha256=package["package_sha256"]
    )
    _append_event(
        ledger,
        state="adjudication_submitted",
        package_sha256=package["package_sha256"],
        result_sha256=result["result_sha256"],
    )
    _append_event(
        ledger,
        state="validated",
        package_sha256=package["package_sha256"],
        result_sha256=result["result_sha256"],
    )
    verify_event_chain(ledger)
    write_object(ledger_path, ledger)
    return _status("validated", len(expected))


def _write_precision_inputs(
    base: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
    result: Mapping[str, Any],
    operator_mapping: Mapping[str, Any],
    submissions: Mapping[str, tuple[Path, Path]],
) -> tuple[dict[str, Path], Mapping[str, Any]]:
    submission_protocol = load_submission_protocol(
        repository_root / protocol["bindings"]["submission_intake"]["path"],
        repository_root=repository_root,
    )
    delivery_protocol = load_delivery_protocol(
        repository_root / submission_protocol["bindings"]["delivery"]["path"],
        repository_root,
    )
    package_root = repository_root / "benchmark/human_annotation_delivery_v1_release"
    recovered = ingest_delivery(
        delivery_protocol,
        package_root=package_root,
        annotator_a=submissions["annotator_a"][0],
        annotator_b=submissions["annotator_b"][0],
    )["recovered"]
    precision_protocol = load_precision_protocol(
        repository_root / protocol["bindings"]["human_precision_adjudication"]["path"],
        repository_root=repository_root,
    )
    context = validate_package(precision_protocol, repository_root=repository_root)
    refs = {"current": context.reference, "prior": context.prior_reference}
    paths = {key: base / f"{key}.json" for key in ("one", "two", "adjudication", "prior")}
    principals = {
        "A": "anon-locked-annotator-A",
        "B": "anon-locked-annotator-B",
        "J": "anon-locked-adjudicator",
    }
    current_maps: dict[str, dict[str, str]] = {}
    for side, round_id, name in (
        ("A", "independent_1", "one"),
        ("B", "independent_2", "two"),
    ):
        submission = IndependentSubmission(
            contract="human_precision_adjudication_v1",
            package=refs["current"],
            round_id=round_id,
            annotator_id=principals[side],
            labels=[LabelRow(**row) for row in recovered[side]["current"]],
        )
        current_maps[side] = {row.item_id: row.label for row in submission.labels}
        write_object(paths[name], submission.model_dump(mode="json"))
    mapping = {
        row["disagreement_alias"]: (row["package_role"], row["item_id"])
        for row in operator_mapping["entries"]
    }
    decisions = {row["disagreement_alias"]: row for row in result["decisions"]}
    current_decisions = [
        AdjudicationRow(
            item_id=item_id,
            final_label=decisions[alias]["final_label"],
            rationale=decisions[alias]["rationale"],
        )
        for alias, (role, item_id) in sorted(mapping.items())
        if role == "current"
    ]
    adjudication = AdjudicationSubmission(
        contract="human_precision_adjudication_v1",
        package=refs["current"],
        round_id="adjudication",
        adjudicator_id=principals["J"],
        decisions=current_decisions,
    )
    write_object(paths["adjudication"], adjudication.model_dump(mode="json"))
    prior_maps = {
        side: {row["item_id"]: row["label"] for row in recovered[side]["prior"]}
        for side in ("A", "B")
    }
    prior_rows: list[PriorLabelRow] = []
    reverse = {
        item_id: decisions[alias]
        for alias, (role, item_id) in mapping.items()
        if role == "prior"
    }
    for item_id in context.required_prior_item_ids:
        a_label, b_label = prior_maps["A"][item_id], prior_maps["B"][item_id]
        if a_label == b_label:
            final_label, resolution = a_label, "annotator_agreement"
        else:
            if item_id not in reverse:
                raise HumanAdjudicationActivationError("prior_adjudication_missing")
            final_label, resolution = reverse[item_id]["final_label"], "adjudicated"
        prior_rows.append(
            PriorLabelRow(
                item_id=item_id,
                annotator_1_label=a_label,
                annotator_2_label=b_label,
                final_label=final_label,
                resolution=resolution,
            )
        )
    prior = PriorResolvedSubmission(
        contract="human_precision_adjudication_v1",
        package=refs["prior"],
        round_id="resolved_prior_package",
        annotator_1_id=principals["A"],
        annotator_2_id=principals["B"],
        adjudicator_id=principals["J"],
        labels=prior_rows,
    )
    write_object(paths["prior"], prior.model_dump(mode="json"))
    return paths, precision_protocol


def unlock_statistics(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    package_path: Path,
    acknowledgement_path: Path,
    result_path: Path,
    ledger_path: Path,
    submission_ledger_path: Path,
    operator_mapping_path: Path,
    submissions: Mapping[str, tuple[Path, Path]],
) -> dict[str, Any]:
    package = verify_package(package_path, protocol)
    _verify_submission_ledger_active(submission_ledger_path, package)
    _verify_acknowledgement(acknowledgement_path, package)
    result = _validate_locked_result(result_path, package)
    ledger = read_object(ledger_path)
    if verify_event_chain(ledger) != "validated":
        raise HumanAdjudicationActivationNotReady("validated_adjudication_required")
    if (
        ledger["package_sha256"] != package["package_sha256"]
        or ledger["queue_sha256"] != package["queue_sha256"]
        or ledger["adjudicator_principal_id"]
        != package["adjudicator_principal_id"]
        or ledger["events"][-1]["result_sha256"] != result["result_sha256"]
    ):
        raise HumanAdjudicationActivationError(
            "validated_result_ledger_binding_mismatch"
        )
    mapping = read_object(operator_mapping_path)
    with tempfile.TemporaryDirectory(prefix="synthetic-human-adjudication-score-") as name:
        paths, precision_protocol = _write_precision_inputs(
            Path(name),
            repository_root,
            protocol,
            package=package,
            result=result,
            operator_mapping=mapping,
            submissions=submissions,
        )
        gate = run_human_precision_gate(
            precision_protocol,
            repository_root=repository_root,
            annotator_one_path=paths["one"],
            annotator_two_path=paths["two"],
            adjudication_path=paths["adjudication"],
            prior_resolved_path=paths["prior"],
        )
    if gate["state"] != "validated":
        raise HumanAdjudicationActivationError("existing_precision_gate_not_validated")
    if (
        gate["coverage"]["expected_independent_item_count"] != 439
        or gate["coverage"]["expected_prior_resolved_item_count"] != 32
    ):
        raise HumanAdjudicationActivationError("complete_chain_count_mismatch")
    _append_event(
        ledger,
        state="statistics_eligible",
        package_sha256=package["package_sha256"],
        result_sha256=result["result_sha256"],
    )
    verify_event_chain(ledger)
    write_object(ledger_path, ledger)
    return {
        "absolute_precision_at_20": None,
        "absolute_precision_reason": "unsupported_from_change_only_package",
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "official_result": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "state": "statistics_eligible",
        "statistics": gate["statistics"],
        "statistics_scope": "human_internal_non_official",
        "status": "adjudication_chain_ready",
        "synthetic_only": package["synthetic_only"],
    }


def _status(state: str, item_count: int) -> dict[str, Any]:
    return {
        "adjudication_item_count": item_count,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "statistics": None,
        "status": "adjudication_chain_ready",
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
            "real_adjudication_queue_missing",
            "real_adjudication_submission_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": "not_ready_missing_real_labels_or_adjudication",
        "statistics": None,
        "status": "not_ready_missing_real_labels_or_adjudication",
    }


def _prepare_synthetic(
    root: Path,
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    no_disagreement: bool = False,
) -> dict[str, Any]:
    submission_protocol = load_submission_protocol(
        repository_root / protocol["bindings"]["submission_intake"]["path"],
        repository_root=repository_root,
    )
    context, bundles, assignment_receipts, assignment_ledger, submissions = (
        _scenario_inputs(root, repository_root, submission_protocol)
    )
    if no_disagreement:
        delivery_root = (
            repository_root / "benchmark/human_annotation_delivery_v1_release"
        )
        delivery_mapping = read_object(delivery_root / "operator/mapping.json")
        identity_by_alias = {
            (row["side"], row["alias"]): (row["role"], row["item_id"])
            for row in delivery_mapping["items"]
        }
        a_value = read_object(submissions["annotator_a"][0])
        b_value = read_object(submissions["annotator_b"][0])
        label_by_identity = {
            identity_by_alias[("A", row["alias"])]: row["label"]
            for row in a_value["labels"]
        }
        for row in b_value["labels"]:
            row["label"] = label_by_identity[identity_by_alias[("B", row["alias"])]]
        b_value["labels_sha256"] = submission_hash(b_value)
        write_object(submissions["annotator_b"][0], b_value)
        build_submission_receipt(
            submissions["annotator_b"][0],
            role="annotator_b",
            assignment_context=context,
            protocol=submission_protocol,
            repository_root=repository_root,
            output=submissions["annotator_b"][1],
        )
    intake_ledger = root / "submission-ledger.json"
    intake_submissions(
        repository_root,
        submission_protocol,
        assignment_context=context,
        submissions=submissions,
        ledger_path=intake_ledger,
        allow_synthetic=True,
    )
    queue_path = root / "queue.json"
    mapping_path = root / "mapping.json"
    build_adjudication_queue(
        repository_root,
        submission_protocol,
        assignment_context=context,
        submissions=submissions,
        ledger_path=intake_ledger,
        queue_path=queue_path,
        operator_mapping_path=mapping_path,
        allow_synthetic=True,
    )
    package_path = root / "package.json"
    activation_ledger = root / "activation-ledger.json"
    issue_package(
        repository_root,
        protocol,
        assignment_context=context,
        submissions=submissions,
        submission_ledger_path=intake_ledger,
        queue_path=queue_path,
        operator_mapping_path=mapping_path,
        challenge=sha256_bytes(b"synthetic-adjudication-challenge"),
        package_path=package_path,
        ledger_path=activation_ledger,
        allow_synthetic=True,
    )
    ack_path = root / "ack.json"
    result_path = root / "result.json"
    build_acknowledgement(package_path, ack_path)
    build_synthetic_result(package_path, result_path)
    return {
        "ack": ack_path,
        "activation_ledger": activation_ledger,
        "assignment_context": context,
        "assignment_ledger": assignment_ledger,
        "assignment_receipts": assignment_receipts,
        "bundles": bundles,
        "mapping": mapping_path,
        "package": package_path,
        "queue": queue_path,
        "result": result_path,
        "submission_ledger": intake_ledger,
        "submission_protocol": submission_protocol,
        "submissions": submissions,
    }


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="synthetic-human-adjudication-matrix-"
    ) as name:
        base = Path(name)
        for scenario in protocol["synthetic_scenarios"]:
            root = base / scenario
            root.mkdir()
            expected = (
                "passed"
                if scenario
                in {
                    "no_disagreement",
                    "valid_multiple_disagreements",
                    "complete_statistics_unlock",
                }
                else "not_ready"
                if scenario == "partial_independent_submissions"
                else "rejected"
            )
            observed, reason = "passed", None
            try:
                if scenario == "partial_independent_submissions":
                    raise HumanAdjudicationActivationNotReady(
                        "two_complete_independent_submissions_required"
                    )
                fixture = _prepare_synthetic(
                    root,
                    repository_root,
                    protocol,
                    no_disagreement=scenario == "no_disagreement",
                )
                if scenario == "no_disagreement" and read_object(
                    fixture["package"]
                )["item_count"] != 0:
                    raise HumanAdjudicationActivationError(
                        "no_disagreement_population_invalid"
                    )
                result = read_object(fixture["result"])
                if scenario == "missing_disagreement" and result["decisions"]:
                    result["decisions"] = result["decisions"][:-1]
                elif scenario in {"extra_disagreement", "forged_disagreement"}:
                    result["decisions"].append(
                        {
                            "disagreement_alias": "disagreement-" + "f" * 24,
                            "final_label": LABELS[0],
                            "rationale": "synthetic extra",
                        }
                    )
                elif scenario == "illegal_label" and result["decisions"]:
                    result["decisions"][0]["final_label"] = "illegal"
                elif scenario == "wrong_adjudicator":
                    result["adjudicator_principal_id"] = "prn_ffffffffffffffff"
                elif scenario == "post_lock_tamper" and result["decisions"]:
                    result["decisions"][0]["rationale"] = "tampered"
                    write_object(fixture["result"], result)
                elif scenario == "old_commit":
                    result["source_commit"] = "0" * 40
                elif scenario == "revoked_upstream":
                    ledger = read_object(fixture["submission_ledger"])
                    _append_submission_event(
                        ledger,
                        state="revoked",
                        role=None,
                        binding=None,
                        queue_sha256=ledger["queue"]["queue_sha256"],
                    )
                    write_object(fixture["submission_ledger"], ledger)
                if scenario != "post_lock_tamper":
                    result["decisions_sha256"] = sha256_bytes(
                        canonical_json(result["decisions"])
                    )
                    result["result_sha256"] = _digest_without(
                        result, "result_sha256"
                    )
                    write_object(fixture["result"], result)
                verify_result(
                    fixture["package"],
                    fixture["ack"],
                    fixture["result"],
                    fixture["activation_ledger"],
                    fixture["submission_ledger"],
                    protocol,
                )
                if scenario == "duplicate_submission":
                    verify_result(
                        fixture["package"],
                        fixture["ack"],
                        fixture["result"],
                        fixture["activation_ledger"],
                        fixture["submission_ledger"],
                        protocol,
                    )
                if scenario == "complete_statistics_unlock":
                    unlocked = unlock_statistics(
                        repository_root,
                        protocol,
                        package_path=fixture["package"],
                        acknowledgement_path=fixture["ack"],
                        result_path=fixture["result"],
                        ledger_path=fixture["activation_ledger"],
                        submission_ledger_path=fixture["submission_ledger"],
                        operator_mapping_path=fixture["mapping"],
                        submissions=fixture["submissions"],
                    )
                    if (
                        unlocked["state"] != "statistics_eligible"
                        or unlocked["statistics_scope"]
                        != "human_internal_non_official"
                        or unlocked["absolute_precision_at_20"] is not None
                    ):
                        raise HumanAdjudicationActivationError(
                            "statistics_unlock_invalid"
                        )
            except HumanAdjudicationActivationNotReady as exc:
                observed, reason = "not_ready", str(exc)
            except (
                HumanAdjudicationActivationError,
                HumanAnnotationSubmissionError,
            ) as exc:
                observed, reason = "rejected", str(exc)
            rows.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "reason": reason,
                    "scenario": scenario,
                }
            )
    if any(row["expected"] != row["observed"] for row in rows):
        raise HumanAdjudicationActivationError("synthetic_matrix_mismatch")
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "passed_scenario_count": len(rows),
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "rows": rows,
        "schema_version": SCHEMA_VERSION,
        "statistics": None,
        "status": "adjudication_chain_ready",
        "synthetic_artifacts_persisted": False,
    }
