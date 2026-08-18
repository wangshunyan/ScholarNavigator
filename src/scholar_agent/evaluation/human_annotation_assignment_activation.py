"""Offline activation gate for real human annotation assignments.

The gate binds already-qualified opaque principals to the frozen A/B delivery
packages and to an adjudicator-only rubric bundle. It never creates labels,
opens the operator mapping, or authenticates a natural person's identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.human_annotation_delivery import (
    load_delivery_protocol,
    verify_delivery,
)


PROTOCOL = "human_annotation_assignment_activation_v1"
BUNDLE_PROTOCOL = "human_annotation_assignment_bundle_v1"
RECEIPT_PROTOCOL = "human_annotation_assignment_receipt_v1"
LEDGER_PROTOCOL = "human_annotation_assignment_event_ledger_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "80cd4bf6f5263231a34a3ad535759f6c6910e835"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ROLES = ("annotator_a", "annotator_b", "adjudicator")
ROLE_TO_SIDE = {"annotator_a": "A", "annotator_b": "B"}
STATES = (
    "prepared",
    "assigned",
    "issued",
    "acknowledged",
    "locked_for_submission",
    "revoked",
    "invalid",
)
TRANSITIONS = {
    "prepared": {"assigned", "revoked", "invalid"},
    "assigned": {"issued", "revoked", "invalid"},
    "issued": {"acknowledged", "revoked", "invalid"},
    "acknowledged": {"locked_for_submission", "revoked", "invalid"},
    "locked_for_submission": {"revoked", "invalid"},
}
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRINCIPAL_RE = re.compile(r"^prn_[0-9a-f]{16}$")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_MEMBER_COUNT = 16
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
FORBIDDEN_BUNDLE_KEYS = frozenset(
    {
        "arm",
        "case_id",
        "global_opaque_id",
        "gold",
        "operator_mapping",
        "private_mapping",
        "qrels",
        "rank",
        "score",
        "source",
        "strategy",
        "target_paper",
    }
)


class HumanAnnotationAssignmentError(RuntimeError):
    """Assignment, package, receipt, or event-chain integrity is invalid."""


class HumanAnnotationAssignmentNotReady(HumanAnnotationAssignmentError):
    """Real qualifications or acknowledgements are incomplete."""


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def decode_object(raw: bytes, *, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HumanAnnotationAssignmentError(reason) from exc
    if not isinstance(value, dict):
        raise HumanAnnotationAssignmentError(reason)
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HumanAnnotationAssignmentError("json_input_unavailable") from exc
    if len(raw) > MAX_MEMBER_BYTES:
        raise HumanAnnotationAssignmentError("json_input_too_large")
    return decode_object(raw, reason="json_input_invalid")


def write_object(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload[key] = ZERO_SHA256
    return sha256_bytes(canonical_json(payload))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise HumanAnnotationAssignmentError("unsafe_path")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.as_posix() != value
        or path.name == ".env"
        or path.parts[0] in {"operator", "third_party"}
    ):
        raise HumanAnnotationAssignmentError("unsafe_path")
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
    value = read_object(path)
    expected_keys = {
        "bindings",
        "execution",
        "formal_validation_complete",
        "issuance",
        "protocol",
        "protocol_sha256",
        "roles",
        "schema_version",
        "source_commit",
        "state_machine",
        "synthetic_scenarios",
    }
    if set(value) != expected_keys:
        raise HumanAnnotationAssignmentError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["roles"] != list(ROLES)
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or _digest_without(value, "protocol_sha256") != value["protocol_sha256"]
    ):
        raise HumanAnnotationAssignmentError("protocol_binding_invalid")
    binding_names = {
        "adjudication",
        "clearance",
        "delivery",
        "delivery_bundle",
        "preregistration_seal",
        "qualification",
        "quarantine",
        "separation_of_duties",
    }
    if not isinstance(value["bindings"], dict) or set(value["bindings"]) != binding_names:
        raise HumanAnnotationAssignmentError("protocol_binding_inventory_invalid")
    for name, spec in value["bindings"].items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise HumanAnnotationAssignmentError("protocol_binding_schema_invalid")
        relative = _safe_relative(spec["path"])
        target = repository_root / relative
        if not target.is_file() or sha256_file(target) != spec["sha256"]:
            raise HumanAnnotationAssignmentError(f"protocol_binding_drift:{name}")
    if value["state_machine"] != {
        "states": list(STATES),
        "transitions": {
            "acknowledged": [
                "locked_for_submission",
                "revoked",
                "invalid",
            ],
            "assigned": ["issued", "revoked", "invalid"],
            "issued": ["acknowledged", "revoked", "invalid"],
            "locked_for_submission": ["revoked", "invalid"],
            "prepared": ["assigned", "revoked", "invalid"],
        },
    }:
        raise HumanAnnotationAssignmentError("state_machine_drift")
    if value["issuance"] != {
        "acknowledgement_required_roles": list(ROLES),
        "adjudicator_payload": "rubric_and_future_disagreement_view_contract_only",
        "alias_packages": {"annotator_a": "A", "annotator_b": "B"},
        "hash_identity_authentication": False,
        "item_count_per_annotator": 471,
        "labels_before_all_acknowledgements": "forbidden",
        "operator_mapping": "excluded",
        "semantic_drift": "invalidate_and_reissue_without_label_migration",
    }:
        raise HumanAnnotationAssignmentError("issuance_contract_drift")
    expected_scenarios = [
        "valid_three_role_issue",
        "annotator_package_swap",
        "shared_principal",
        "adjudicator_early_unblinding",
        "duplicate_claim",
        "package_tamper",
        "revoked_qualification",
        "expired_qualification",
        "commit_drift",
        "partial_acknowledgement",
        "post_issue_protocol_change",
        "coordinator_claim",
    ]
    if value["synthetic_scenarios"] != expected_scenarios:
        raise HumanAnnotationAssignmentError("synthetic_scenario_contract_drift")
    return value


def _validate_qualifications(
    qualifications: Sequence[Mapping[str, Any]], *, allow_synthetic: bool
) -> dict[str, Mapping[str, Any]]:
    by_role: dict[str, Mapping[str, Any]] = {}
    for row in qualifications:
        role = row.get("role")
        if role not in ROLES or role in by_role:
            raise HumanAnnotationAssignmentError("qualification_role_population_invalid")
        if (
            row.get("submission_protocol")
            != "human_annotator_qualification_submission_v1"
            or row.get("lifecycle") != "active"
            or row.get("source_commit")
            != "fdd0f8744d99c5802e867db63b0c0ee032972e09"
            or row.get("valid_for_source_commit")
            != "fdd0f8744d99c5802e867db63b0c0ee032972e09"
            or row.get("rubric_acknowledged") is not True
            or row.get("calibration", {}).get("summary", {}).get("qualified")
            is not True
            or not isinstance(row.get("qualification_sha256"), str)
            or not SHA256_RE.fullmatch(str(row["qualification_sha256"]))
            or _digest_without(row, "qualification_sha256")
            != row["qualification_sha256"]
        ):
            raise HumanAnnotationAssignmentError("qualification_not_fresh")
        if row.get("synthetic_principal") is not False and not (
            allow_synthetic and row.get("synthetic_principal") is True
        ):
            raise HumanAnnotationAssignmentError("real_qualification_required")
        if (
            not isinstance(row.get("principal_id"), str)
            or not PRINCIPAL_RE.fullmatch(str(row["principal_id"]))
            or not isinstance(row.get("principal_commitment"), str)
            or not SHA256_RE.fullmatch(str(row["principal_commitment"]))
        ):
            raise HumanAnnotationAssignmentError("principal_binding_invalid")
        by_role[str(role)] = row
    if set(by_role) != set(ROLES):
        raise HumanAnnotationAssignmentNotReady("qualification_population_incomplete")
    principals = [str(by_role[role]["principal_id"]) for role in ROLES]
    commitments = [str(by_role[role]["principal_commitment"]) for role in ROLES]
    if len(set(principals)) != 3:
        raise HumanAnnotationAssignmentError("principal_role_conflict")
    if len(set(commitments)) != 3:
        raise HumanAnnotationAssignmentError("principal_alias_rebinding")
    return by_role


def _delivery_state(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    package_root = repository_root / "benchmark/human_annotation_delivery_v1_release"
    delivery_protocol = load_delivery_protocol(
        repository_root / protocol["bindings"]["delivery"]["path"],
        repository_root,
    )
    verified = verify_delivery(delivery_protocol, package_root)
    if verified["item_count_per_annotator"] != 471:
        raise HumanAnnotationAssignmentError("delivery_population_invalid")
    bundle = read_object(package_root / "bundle.json")
    rows: dict[str, list[dict[str, Any]]] = {}
    alias_sets: dict[str, set[str]] = {}
    for side in ("A", "B"):
        try:
            value = json.loads(
                (package_root / f"annotator-{side}/items.json").read_text(
                    encoding="utf-8"
                ),
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise HumanAnnotationAssignmentError("delivery_items_invalid") from exc
        if not isinstance(value, list) or len(value) != 471:
            raise HumanAnnotationAssignmentError("delivery_items_invalid")
        aliases: set[str] = set()
        for item in value:
            if (
                not isinstance(item, dict)
                or set(item) != {"abstract", "alias", "query", "title", "year"}
                or not isinstance(item["alias"], str)
                or item["alias"] in aliases
            ):
                raise HumanAnnotationAssignmentError("delivery_items_invalid")
            aliases.add(item["alias"])
        rows[side] = value
        alias_sets[side] = aliases
    if alias_sets["A"] & alias_sets["B"]:
        raise HumanAnnotationAssignmentError("delivery_alias_sets_overlap")
    return package_root, bundle, rows


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _receipt_template(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "acknowledged": False,
        "assignment_challenge": manifest["assignment_challenge"],
        "assignment_protocol_sha256": manifest["assignment_protocol_sha256"],
        "bundle_sha256": ZERO_SHA256,
        "principal_id": manifest["principal_id"],
        "receipt_protocol": RECEIPT_PROTOCOL,
        "receipt_sha256": ZERO_SHA256,
        "role": manifest["role"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "state": "issued",
        "submitted_by_role": "principal_self",
    }


def _build_role_bundle(
    repository_root: Path,
    protocol: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    challenge: str,
    output: Path,
    package_side_override: str | None = None,
    adjudicator_unblind: bool = False,
) -> None:
    if not isinstance(challenge, str) or not SHA256_RE.fullmatch(challenge):
        raise HumanAnnotationAssignmentError("assignment_challenge_invalid")
    package_root, delivery_bundle, items = _delivery_state(repository_root, protocol)
    role = str(qualification["role"])
    files: dict[str, bytes] = {
        "README.txt": (
            "One-time offline blind annotation assignment. Hashes prove content "
            "integrity, not the natural person's identity. Do not forward or "
            "reuse this role-bound bundle.\n"
        ).encode("utf-8"),
        "verify.py": (
            repository_root / "scripts/human_annotation_assignment_runtime.py"
        ).read_bytes(),
    }
    package_identity: dict[str, Any]
    item_count: int
    if role in ROLE_TO_SIDE:
        expected_side = ROLE_TO_SIDE[role]
        side = package_side_override or expected_side
        source_dir = package_root / f"annotator-{side}"
        names = (
            "app.js",
            "index.html",
            "items-data.js",
            "items.json",
            "package-data.js",
            "package.json",
            "rubric.json",
        )
        for name in names:
            files[f"payload/{name}"] = (source_dir / name).read_bytes()
        package = read_object(source_dir / "package.json")
        aliases = sorted(str(row["alias"]) for row in items[side])
        package_identity = {
            "alias_set_sha256": sha256_bytes(canonical_json(aliases)),
            "delivery_bundle_sha256": delivery_bundle["bundle_sha256"],
            "package_id": package["package_id"],
            "package_sha256": package["package_sha256"],
            "side": side,
        }
        item_count = 471
    else:
        rubric = (package_root / "annotator-A/rubric.json").read_bytes()
        disagreement_contract = {
            "allowed_fields": [
                "disagreement_alias",
                "annotation_a",
                "annotation_b",
                "rubric",
            ],
            "available_after_independent_lock": True,
            "cross_package_mapping": "forbidden",
            "original_annotation_access_before_lock": False,
            "protocol": "future_disagreement_view_v1",
            "schema_version": SCHEMA_VERSION,
        }
        files["rubric.json"] = rubric
        files["disagreement_view_contract.json"] = canonical_json(
            disagreement_contract
        )
        if adjudicator_unblind:
            files["payload/items.json"] = (
                package_root / "annotator-A/items.json"
            ).read_bytes()
        package_identity = {
            "delivery_bundle_sha256": delivery_bundle["bundle_sha256"],
            "future_disagreement_contract_sha256": sha256_bytes(
                canonical_json(disagreement_contract)
            ),
            "rubric_sha256": sha256_bytes(rubric),
            "side": "adjudicator_only",
        }
        item_count = 0
    manifest: dict[str, Any] = {
        "assignment_challenge": challenge,
        "assignment_protocol_sha256": protocol["protocol_sha256"],
        "bundle_protocol": BUNDLE_PROTOCOL,
        "files": [],
        "formal_validation_complete": False,
        "item_count": item_count,
        "manifest_sha256": ZERO_SHA256,
        "package_identity": package_identity,
        "principal_commitment": qualification["principal_commitment"],
        "principal_id": qualification["principal_id"],
        "qualification_sha256": qualification["qualification_sha256"],
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "state": "issued",
        "synthetic_only": bool(qualification["synthetic_principal"]),
    }
    files["receipt_template.json"] = canonical_json(_receipt_template(manifest))
    manifest["files"] = [
        {"path": name, "sha256": sha256_bytes(raw), "size": len(raw)}
        for name, raw in sorted(files.items())
    ]
    manifest["manifest_sha256"] = _digest_without(manifest, "manifest_sha256")
    files["manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(_zip_info(name), raw)


def _read_archive(path: Path) -> dict[str, bytes]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise HumanAnnotationAssignmentError("bundle_size_or_presence_invalid")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise HumanAnnotationAssignmentError("bundle_member_limit")
            files: dict[str, bytes] = {}
            for info in infos:
                name = _safe_relative(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    name in files
                    or info.is_dir()
                    or mode not in (0, 0o100000)
                    or info.file_size > MAX_MEMBER_BYTES
                ):
                    raise HumanAnnotationAssignmentError("bundle_member_unsafe")
                files[name] = archive.read(info)
    except HumanAnnotationAssignmentError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise HumanAnnotationAssignmentError("bundle_invalid") from exc
    return files


def verify_bundle(
    path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    expected_role: str | None = None,
) -> dict[str, Any]:
    files = _read_archive(path)
    if "manifest.json" not in files:
        raise HumanAnnotationAssignmentError("bundle_manifest_missing")
    manifest = decode_object(files["manifest.json"], reason="bundle_manifest_invalid")
    expected_keys = {
        "assignment_challenge",
        "assignment_protocol_sha256",
        "bundle_protocol",
        "files",
        "formal_validation_complete",
        "item_count",
        "manifest_sha256",
        "package_identity",
        "principal_commitment",
        "principal_id",
        "qualification_sha256",
        "role",
        "schema_version",
        "source_commit",
        "state",
        "synthetic_only",
    }
    if set(manifest) != expected_keys:
        raise HumanAnnotationAssignmentError("bundle_manifest_schema_invalid")
    role = manifest["role"]
    if (
        manifest["bundle_protocol"] != BUNDLE_PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["assignment_protocol_sha256"] != protocol["protocol_sha256"]
        or manifest["state"] != "issued"
        or manifest["formal_validation_complete"] is not False
        or role not in ROLES
        or (expected_role is not None and role != expected_role)
        or _digest_without(manifest, "manifest_sha256")
        != manifest["manifest_sha256"]
    ):
        raise HumanAnnotationAssignmentError("bundle_binding_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise HumanAnnotationAssignmentError("bundle_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or row["path"] in seen
            or row["path"] == "manifest.json"
            or row["path"] not in files
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != sha256_bytes(files[row["path"]])
        ):
            raise HumanAnnotationAssignmentError("bundle_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise HumanAnnotationAssignmentError("bundle_inventory_invalid")
    if "verify.py" not in files or files["verify.py"] != (
        repository_root / "scripts/human_annotation_assignment_runtime.py"
    ).read_bytes():
        raise HumanAnnotationAssignmentError("bundle_runtime_drift")
    if role in ROLE_TO_SIDE:
        expected_payload = {
            "README.txt",
            "payload/app.js",
            "payload/index.html",
            "payload/items-data.js",
            "payload/items.json",
            "payload/package-data.js",
            "payload/package.json",
            "payload/rubric.json",
            "receipt_template.json",
            "verify.py",
        }
        if set(files) - {"manifest.json"} != expected_payload:
            raise HumanAnnotationAssignmentError("annotator_payload_invalid")
        package = decode_object(
            files["payload/package.json"], reason="annotator_package_invalid"
        )
        side = ROLE_TO_SIDE[str(role)]
        if (
            package.get("side") != side
            or manifest["package_identity"].get("side") != side
            or manifest["item_count"] != 471
        ):
            raise HumanAnnotationAssignmentError("annotator_package_role_mismatch")
        frozen_root = (
            repository_root
            / "benchmark/human_annotation_delivery_v1_release"
            / f"annotator-{side}"
        )
        for name in (
            "app.js",
            "index.html",
            "items-data.js",
            "items.json",
            "package-data.js",
            "package.json",
            "rubric.json",
        ):
            if files[f"payload/{name}"] != (frozen_root / name).read_bytes():
                raise HumanAnnotationAssignmentError(
                    "annotator_frozen_package_drift"
                )
        try:
            items = json.loads(
                files["payload/items.json"].decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise HumanAnnotationAssignmentError("annotator_items_invalid") from exc
        if not isinstance(items, list) or len(items) != 471:
            raise HumanAnnotationAssignmentError("annotator_items_invalid")
        aliases: list[str] = []
        for item in items:
            if (
                not isinstance(item, dict)
                or set(item) != {"abstract", "alias", "query", "title", "year"}
                or not isinstance(item["alias"], str)
            ):
                raise HumanAnnotationAssignmentError("annotator_items_invalid")
            aliases.append(item["alias"])
        if (
            len(set(aliases)) != 471
            or manifest["package_identity"].get("alias_set_sha256")
            != sha256_bytes(canonical_json(sorted(aliases)))
        ):
            raise HumanAnnotationAssignmentError("annotator_alias_binding_invalid")
    else:
        expected_payload = {
            "README.txt",
            "disagreement_view_contract.json",
            "receipt_template.json",
            "rubric.json",
            "verify.py",
        }
        if (
            set(files) - {"manifest.json"} != expected_payload
            or manifest["item_count"] != 0
            or manifest["package_identity"].get("side") != "adjudicator_only"
            or files["rubric.json"]
            != (
                repository_root
                / "benchmark/human_annotation_delivery_v1_release"
                / "annotator-A/rubric.json"
            ).read_bytes()
        ):
            raise HumanAnnotationAssignmentError("adjudicator_blinding_violation")
    if _walk_keys(
        {
            "manifest": manifest,
            "receipt_template": decode_object(
                files["receipt_template.json"], reason="receipt_template_invalid"
            ),
        }
    ) & FORBIDDEN_BUNDLE_KEYS:
        raise HumanAnnotationAssignmentError("bundle_forbidden_metadata")
    return manifest


def _event(
    manifest: Mapping[str, Any],
    *,
    state: str,
    previous_sha256: str,
    sequence: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "assignment_challenge": manifest["assignment_challenge"],
        "bundle_sha256": manifest["bundle_sha256"],
        "event_sha256": ZERO_SHA256,
        "previous_sha256": previous_sha256,
        "principal_id": manifest["principal_id"],
        "qualification_sha256": manifest["qualification_sha256"],
        "role": manifest["role"],
        "sequence": sequence,
        "source_commit": SOURCE_COMMIT,
        "state": state,
    }
    value["event_sha256"] = _digest_without(value, "event_sha256")
    return value


def _append_state(
    ledger: dict[str, Any], manifest: Mapping[str, Any], state: str
) -> None:
    previous = (
        ledger["events"][-1]["event_sha256"]
        if ledger["events"]
        else ZERO_SHA256
    )
    ledger["events"].append(
        _event(
            manifest,
            state=state,
            previous_sha256=previous,
            sequence=len(ledger["events"]) + 1,
        )
    )


def verify_event_chain(ledger: Mapping[str, Any]) -> dict[str, str]:
    if set(ledger) != {
        "events",
        "formal_validation_complete",
        "protocol",
        "receipts",
        "source_commit",
    } or (
        ledger["protocol"] != LEDGER_PROTOCOL
        or ledger["source_commit"] != SOURCE_COMMIT
        or ledger["formal_validation_complete"] is not False
        or not isinstance(ledger["events"], list)
        or not isinstance(ledger["receipts"], list)
    ):
        raise HumanAnnotationAssignmentError("assignment_ledger_invalid")
    previous = ZERO_SHA256
    states: dict[str, str] = {}
    bindings: dict[str, tuple[Any, ...]] = {}
    for sequence, event in enumerate(ledger["events"], 1):
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "assignment_challenge",
                "bundle_sha256",
                "event_sha256",
                "previous_sha256",
                "principal_id",
                "qualification_sha256",
                "role",
                "sequence",
                "source_commit",
                "state",
            }
            or event["sequence"] != sequence
            or event["previous_sha256"] != previous
            or event["source_commit"] != SOURCE_COMMIT
            or event["role"] not in ROLES
            or event["state"] not in STATES
            or _digest_without(event, "event_sha256") != event["event_sha256"]
        ):
            raise HumanAnnotationAssignmentError("assignment_event_chain_invalid")
        role = str(event["role"])
        binding = (
            event["principal_id"],
            event["qualification_sha256"],
            event["assignment_challenge"],
            event["bundle_sha256"],
        )
        if role in bindings and bindings[role] != binding:
            raise HumanAnnotationAssignmentError("assignment_event_binding_drift")
        bindings[role] = binding
        prior_state = states.get(role)
        if prior_state is None:
            if event["state"] != "prepared":
                raise HumanAnnotationAssignmentError("assignment_state_invalid")
        elif event["state"] not in TRANSITIONS.get(prior_state, set()):
            raise HumanAnnotationAssignmentError("assignment_transition_invalid")
        states[role] = str(event["state"])
        previous = str(event["event_sha256"])
    receipt_hashes: set[str] = set()
    for receipt in ledger["receipts"]:
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"receipt_sha256", "role"}
            or receipt["role"] not in ROLES
            or not isinstance(receipt["receipt_sha256"], str)
            or not SHA256_RE.fullmatch(receipt["receipt_sha256"])
            or receipt["receipt_sha256"] in receipt_hashes
        ):
            raise HumanAnnotationAssignmentError("assignment_receipt_ledger_invalid")
        receipt_hashes.add(receipt["receipt_sha256"])
    return states


def issue_assignments(
    repository_root: Path,
    protocol: Mapping[str, Any],
    qualifications: Sequence[Mapping[str, Any]],
    *,
    challenges: Mapping[str, str],
    output_root: Path,
    ledger_path: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    by_role = _validate_qualifications(
        qualifications, allow_synthetic=allow_synthetic
    )
    if set(challenges) != set(ROLES) or len(set(challenges.values())) != 3:
        raise HumanAnnotationAssignmentError("assignment_challenge_population_invalid")
    if ledger_path.exists() or output_root.exists():
        raise HumanAnnotationAssignmentError("assignment_target_not_empty")
    output_root.mkdir(parents=True)
    manifests: dict[str, dict[str, Any]] = {}
    bundles: dict[str, str] = {}
    for role in ROLES:
        bundle_path = output_root / f"{role}.zip"
        _build_role_bundle(
            repository_root,
            protocol,
            by_role[role],
            challenge=challenges[role],
            output=bundle_path,
        )
        manifest = verify_bundle(
            bundle_path,
            protocol,
            repository_root=repository_root,
            expected_role=role,
        )
        manifest["bundle_sha256"] = sha256_file(bundle_path)
        manifests[role] = manifest
        bundles[role] = manifest["bundle_sha256"]
    ledger: dict[str, Any] = {
        "events": [],
        "formal_validation_complete": False,
        "protocol": LEDGER_PROTOCOL,
        "receipts": [],
        "source_commit": SOURCE_COMMIT,
    }
    for state in ("prepared", "assigned", "issued"):
        for role in ROLES:
            _append_state(ledger, manifests[role], state)
    verify_event_chain(ledger)
    write_object(ledger_path, ledger)
    return {
        "bundle_sha256": bundles,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "role_count": 3,
        "schema_version": SCHEMA_VERSION,
        "state": "issued",
        "status": "assignment_chain_ready",
    }


def build_acknowledgement(
    bundle_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    output: Path,
    submitted_by_role: str = "principal_self",
) -> None:
    manifest = verify_bundle(
        bundle_path, protocol, repository_root=repository_root
    )
    value: dict[str, Any] = {
        "acknowledged": True,
        "assignment_challenge": manifest["assignment_challenge"],
        "assignment_protocol_sha256": manifest["assignment_protocol_sha256"],
        "bundle_sha256": sha256_file(bundle_path),
        "principal_id": manifest["principal_id"],
        "receipt_protocol": RECEIPT_PROTOCOL,
        "receipt_sha256": ZERO_SHA256,
        "role": manifest["role"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "state": "acknowledged",
        "submitted_by_role": submitted_by_role,
    }
    value["receipt_sha256"] = _digest_without(value, "receipt_sha256")
    write_object(output, value)


def verify_acknowledgements(
    bundle_paths: Mapping[str, Path],
    receipt_paths: Sequence[Path],
    ledger_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    if set(bundle_paths) != set(ROLES):
        raise HumanAnnotationAssignmentError("bundle_population_invalid")
    ledger = read_object(ledger_path)
    states = verify_event_chain(ledger)
    if any(states.get(role) != "issued" for role in ROLES):
        raise HumanAnnotationAssignmentError("assignment_not_issuable")
    manifests: dict[str, dict[str, Any]] = {}
    bundle_hashes: dict[str, str] = {}
    for role in ROLES:
        manifests[role] = verify_bundle(
            bundle_paths[role],
            protocol,
            repository_root=repository_root,
            expected_role=role,
        )
        manifests[role]["bundle_sha256"] = sha256_file(bundle_paths[role])
        bundle_hashes[role] = manifests[role]["bundle_sha256"]
    receipts: dict[str, dict[str, Any]] = {}
    existing = {str(row["receipt_sha256"]) for row in ledger["receipts"]}
    for path in receipt_paths:
        receipt = read_object(path)
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
        role = receipt.get("role")
        if (
            set(receipt) != expected_keys
            or role not in ROLES
            or role in receipts
            or receipt["receipt_protocol"] != RECEIPT_PROTOCOL
            or receipt["schema_version"] != SCHEMA_VERSION
            or receipt["source_commit"] != SOURCE_COMMIT
            or receipt["assignment_protocol_sha256"] != protocol["protocol_sha256"]
            or receipt["acknowledged"] is not True
            or receipt["state"] != "acknowledged"
            or receipt["submitted_by_role"] != "principal_self"
            or _digest_without(receipt, "receipt_sha256")
            != receipt["receipt_sha256"]
            or receipt["receipt_sha256"] in existing
        ):
            raise HumanAnnotationAssignmentError("assignment_receipt_invalid")
        manifest = manifests[str(role)]
        if (
            receipt["principal_id"] != manifest["principal_id"]
            or receipt["assignment_challenge"]
            != manifest["assignment_challenge"]
            or receipt["bundle_sha256"] != bundle_hashes[str(role)]
        ):
            raise HumanAnnotationAssignmentError("assignment_receipt_binding_invalid")
        receipts[str(role)] = receipt
    if set(receipts) != set(ROLES):
        raise HumanAnnotationAssignmentNotReady("acknowledgement_incomplete")
    for role in ROLES:
        _append_state(ledger, manifests[role], "acknowledged")
        ledger["receipts"].append(
            {
                "receipt_sha256": receipts[role]["receipt_sha256"],
                "role": role,
            }
        )
    for role in ROLES:
        _append_state(ledger, manifests[role], "locked_for_submission")
    ledger["receipts"] = sorted(ledger["receipts"], key=lambda row: row["role"])
    final_states = verify_event_chain(ledger)
    if set(final_states.values()) != {"locked_for_submission"}:
        raise HumanAnnotationAssignmentError("assignment_lock_incomplete")
    write_object(ledger_path, ledger)
    return {
        "acknowledged_role_count": 3,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "label_intake_allowed": True,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": "locked_for_submission",
        "status": "assignment_chain_ready",
    }


def label_intake_allowed(ledger_path: Path) -> bool:
    states = verify_event_chain(read_object(ledger_path))
    return set(states) == set(ROLES) and set(states.values()) == {
        "locked_for_submission"
    }


def _tamper_archive(path: Path) -> None:
    files = _read_archive(path)
    files["README.txt"] += b"tampered"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(_zip_info(name), raw)


def simulate_matrix(
    repository_root: Path,
    protocol: Mapping[str, Any],
    qualification_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from scholar_agent.evaluation.human_annotator_qualification_intake import (
        build_contract,
        build_synthetic_submission,
        read_object as read_qualification,
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="human-annotation-assignment-matrix-"
    ) as temp_name:
        root = Path(temp_name)
        base_qualifications: list[dict[str, Any]] = []
        for ordinal, role in enumerate(ROLES, 1):
            contract = build_contract(
                qualification_protocol,
                challenge=sha256_bytes(f"qualification:{role}".encode()),
                role=role,
            )
            submission_path = root / f"qualification-{role}.json"
            build_synthetic_submission(
                contract,
                submission_path,
                principal_id=f"prn_{ordinal:016x}",
                principal_commitment=sha256_bytes(f"principal:{role}".encode()),
            )
            base_qualifications.append(read_qualification(submission_path))
        challenges = {
            role: sha256_bytes(f"assignment:{role}".encode()) for role in ROLES
        }
        for scenario in protocol["synthetic_scenarios"]:
            scenario_root = root / scenario
            qualifications = copy.deepcopy(base_qualifications)
            expected = "passed" if scenario == "valid_three_role_issue" else "rejected"
            observed = "passed"
            reason = None
            try:
                if scenario == "shared_principal":
                    qualifications[1]["principal_id"] = qualifications[0][
                        "principal_id"
                    ]
                    qualifications[1]["qualification_sha256"] = _digest_without(
                        qualifications[1], "qualification_sha256"
                    )
                elif scenario == "revoked_qualification":
                    qualifications[0]["lifecycle"] = "revoked"
                    qualifications[0]["qualification_sha256"] = _digest_without(
                        qualifications[0], "qualification_sha256"
                    )
                elif scenario == "expired_qualification":
                    qualifications[0]["valid_for_source_commit"] = "0" * 40
                    qualifications[0]["qualification_sha256"] = _digest_without(
                        qualifications[0], "qualification_sha256"
                    )
                output_root = scenario_root / "bundles"
                ledger = scenario_root / "ledger.json"
                issue_assignments(
                    repository_root,
                    protocol,
                    qualifications,
                    challenges=challenges,
                    output_root=output_root,
                    ledger_path=ledger,
                    allow_synthetic=True,
                )
                bundle_paths = {
                    role: output_root / f"{role}.zip" for role in ROLES
                }
                if scenario == "annotator_package_swap":
                    _build_role_bundle(
                        repository_root,
                        protocol,
                        qualifications[0],
                        challenge=challenges["annotator_a"],
                        output=bundle_paths["annotator_a"],
                        package_side_override="B",
                    )
                    verify_bundle(
                        bundle_paths["annotator_a"],
                        protocol,
                        repository_root=repository_root,
                        expected_role="annotator_a",
                    )
                elif scenario == "adjudicator_early_unblinding":
                    _build_role_bundle(
                        repository_root,
                        protocol,
                        qualifications[2],
                        challenge=challenges["adjudicator"],
                        output=bundle_paths["adjudicator"],
                        adjudicator_unblind=True,
                    )
                    verify_bundle(
                        bundle_paths["adjudicator"],
                        protocol,
                        repository_root=repository_root,
                        expected_role="adjudicator",
                    )
                elif scenario == "package_tamper":
                    _tamper_archive(bundle_paths["annotator_a"])
                    verify_bundle(
                        bundle_paths["annotator_a"],
                        protocol,
                        repository_root=repository_root,
                    )
                receipts: list[Path] = []
                for role in ROLES:
                    receipt = scenario_root / f"receipt-{role}.json"
                    build_acknowledgement(
                        bundle_paths[role],
                        protocol,
                        repository_root=repository_root,
                        output=receipt,
                        submitted_by_role=(
                            "human_package_coordinator"
                            if scenario == "coordinator_claim" and role == "annotator_a"
                            else "principal_self"
                        ),
                    )
                    receipts.append(receipt)
                if scenario == "commit_drift":
                    value = read_object(receipts[0])
                    value["source_commit"] = "f" * 40
                    value["receipt_sha256"] = _digest_without(
                        value, "receipt_sha256"
                    )
                    write_object(receipts[0], value)
                elif scenario == "post_issue_protocol_change":
                    value = read_object(receipts[0])
                    value["assignment_protocol_sha256"] = "f" * 64
                    value["receipt_sha256"] = _digest_without(
                        value, "receipt_sha256"
                    )
                    write_object(receipts[0], value)
                receipt_input = (
                    receipts[:2]
                    if scenario == "partial_acknowledgement"
                    else receipts
                )
                verify_acknowledgements(
                    bundle_paths,
                    receipt_input,
                    ledger,
                    protocol,
                    repository_root=repository_root,
                )
                if scenario == "duplicate_claim":
                    verify_acknowledgements(
                        bundle_paths,
                        receipts,
                        ledger,
                        protocol,
                        repository_root=repository_root,
                    )
            except HumanAnnotationAssignmentError as exc:
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
        raise HumanAnnotationAssignmentError("synthetic_matrix_mismatch")
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_READY,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "passed_count": len(rows),
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "real_package_distributed": False,
        "scenario_count": len(rows),
        "scenarios": rows,
        "schema_version": SCHEMA_VERSION,
        "status": "assignment_chain_ready",
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
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "label_intake_allowed": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "real_package_distributed": False,
        "schema_version": SCHEMA_VERSION,
        "state": "not_ready_missing_real_qualified_principals",
        "status": "not_ready_missing_real_qualified_principals",
    }
