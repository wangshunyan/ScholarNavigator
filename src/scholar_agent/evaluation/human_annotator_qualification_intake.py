"""Offline intake for real human annotator and adjudicator qualifications.

The gate accepts only opaque principals, structured conflict declarations, and
results from synthetic rubric calibration. It never opens the frozen 471-item
package and never creates a relevance label or Precision value.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "human_annotator_qualification_intake_v1"
KIT_PROTOCOL = "human_annotator_qualification_kit_v1"
SUBMISSION_PROTOCOL = "human_annotator_qualification_submission_v1"
ASSIGNMENT_PROTOCOL = "human_annotator_role_assignment_proposal_v1"
LEDGER_PROTOCOL = "human_annotator_qualification_intake_ledger_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "fdd0f8744d99c5802e867db63b0c0ee032972e09"
EXIT_QUALIFIED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ROLES = ("annotator_a", "annotator_b", "adjudicator")
LABELS = (
    "relevant",
    "partially_relevant",
    "not_relevant",
    "insufficient_information",
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRINCIPAL_RE = re.compile(r"^prn_[0-9a-f]{16}$")
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 1024 * 1024
MAX_MEMBER_COUNT = 8
MAX_NOTES_LENGTH = 160
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "real_label_count": 0,
    "snapshot_write_count": 0,
}
REAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
FORBIDDEN_IDENTITY_KEYS = frozenset(
    {
        "credential",
        "email",
        "employer",
        "hostname",
        "name",
        "organization",
        "real_identity",
        "unit",
        "username",
    }
)
CALIBRATION_ITEMS = (
    {
        "calibration_id": "synthetic-calibration-01",
        "paper_summary": "A synthetic paper directly answers every stated query concept.",
        "query_summary": "Find studies that directly answer all stated concepts.",
    },
    {
        "calibration_id": "synthetic-calibration-02",
        "paper_summary": "A synthetic paper addresses one central concept but omits another.",
        "query_summary": "Find studies addressing two required concepts.",
    },
    {
        "calibration_id": "synthetic-calibration-03",
        "paper_summary": "A synthetic paper concerns an unrelated domain.",
        "query_summary": "Find studies in a different synthetic domain.",
    },
    {
        "calibration_id": "synthetic-calibration-04",
        "paper_summary": "Only a title is available and it is too ambiguous to decide.",
        "query_summary": "Find studies with a specific mechanism and outcome.",
    },
    {
        "calibration_id": "synthetic-calibration-05",
        "paper_summary": "A synthetic paper directly studies the requested mechanism and outcome.",
        "query_summary": "Find studies with the requested mechanism and outcome.",
    },
    {
        "calibration_id": "synthetic-calibration-06",
        "paper_summary": "A synthetic paper is methodologically adjacent but answers another question.",
        "query_summary": "Find studies answering the exact synthetic question.",
    },
)
CALIBRATION_REFERENCE = {
    "synthetic-calibration-01": "relevant",
    "synthetic-calibration-02": "partially_relevant",
    "synthetic-calibration-03": "not_relevant",
    "synthetic-calibration-04": "insufficient_information",
    "synthetic-calibration-05": "relevant",
    "synthetic-calibration-06": "not_relevant",
}


class HumanAnnotatorQualificationError(RuntimeError):
    """Qualification, conflict, challenge, or role binding is invalid."""


class HumanAnnotatorQualificationNotReady(HumanAnnotatorQualificationError):
    """Real qualified principals have not all been imported."""


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
        raise HumanAnnotatorQualificationError(reason) from exc
    if not isinstance(value, dict):
        raise HumanAnnotatorQualificationError(reason)
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HumanAnnotatorQualificationError("json_input_unavailable") from exc
    if len(raw) > MAX_MEMBER_BYTES:
        raise HumanAnnotatorQualificationError("json_input_too_large")
    return decode_object(raw, reason="json_input_invalid")


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload[key] = ZERO_SHA256
    return sha256_bytes(canonical_json(payload))


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise HumanAnnotatorQualificationError("unsafe_path")
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
        raise HumanAnnotatorQualificationError("unsafe_path")
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
    if set(value) != {
        "bindings",
        "calibration",
        "execution",
        "formal_validation_complete",
        "intake_contract",
        "protocol",
        "protocol_sha256",
        "roles",
        "schema_version",
        "source_commit",
        "synthetic_scenarios",
    }:
        raise HumanAnnotatorQualificationError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["roles"] != list(ROLES)
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or _digest_without(value, "protocol_sha256") != value["protocol_sha256"]
    ):
        raise HumanAnnotatorQualificationError("protocol_binding_invalid")
    required_bindings = {
        "adjudication",
        "clearance",
        "delivery",
        "preregistration",
        "quarantine",
        "separation_of_duties",
    }
    if not isinstance(value["bindings"], dict) or set(value["bindings"]) != required_bindings:
        raise HumanAnnotatorQualificationError("protocol_binding_inventory_invalid")
    for name, spec in value["bindings"].items():
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
            raise HumanAnnotatorQualificationError("protocol_binding_schema_invalid")
        target = repository_root / _safe_relative(spec["path"])
        if not target.is_file() or sha256_file(target) != spec["sha256"]:
            raise HumanAnnotatorQualificationError(f"protocol_binding_drift:{name}")
    if value["calibration"] != {
        "allowed_labels": list(LABELS),
        "item_count": len(CALIBRATION_ITEMS),
        "notes_max_length": MAX_NOTES_LENGTH,
        "required_correct": len(CALIBRATION_ITEMS),
        "synthetic_only": True,
    }:
        raise HumanAnnotatorQualificationError("calibration_contract_drift")
    if value["intake_contract"] != {
        "alias_rebinding": "forbidden_by_principal_commitment",
        "challenge": "single_use_role_commit_and_protocol_bound",
        "coordinator_submission": "forbidden",
        "identity_fields": "opaque_principal_and_commitment_only",
        "posthoc_signature": "forbidden",
        "real_package_distribution": False,
        "role_assignment_state": "ready_for_real_assignment_only",
    }:
        raise HumanAnnotatorQualificationError("intake_contract_drift")
    expected = [
        "qualified_three_roles",
        "same_principal",
        "annotator_is_adjudicator",
        "coordinator_proxy",
        "calibration_incomplete",
        "illegal_label",
        "challenge_replay",
        "identity_alias",
        "commit_drift",
        "qualification_tamper",
        "revoked_reuse",
        "expired_qualification",
    ]
    if value["synthetic_scenarios"] != {"names": expected, "synthetic_only": True}:
        raise HumanAnnotatorQualificationError("synthetic_scenario_contract_drift")
    return value


def build_contract(
    protocol: Mapping[str, Any], *, challenge: str, role: str
) -> dict[str, Any]:
    if not isinstance(challenge, str) or not SHA256_RE.fullmatch(challenge):
        raise HumanAnnotatorQualificationError("challenge_invalid")
    if role not in ROLES:
        raise HumanAnnotatorQualificationError("role_invalid")
    value: dict[str, Any] = {
        "bindings": {
            name: protocol["bindings"][name]["sha256"]
            for name in (
                "adjudication",
                "delivery",
                "separation_of_duties",
            )
        },
        "calibration_items_sha256": sha256_bytes(canonical_json(CALIBRATION_ITEMS)),
        "challenge": challenge,
        "forbidden_identity_fields": sorted(FORBIDDEN_IDENTITY_KEYS),
        "kit_protocol": KIT_PROTOCOL,
        "qualification_protocol": SUBMISSION_PROTOCOL,
        "role": role,
        "rubric_protocol_sha256": protocol["bindings"]["adjudication"]["sha256"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    value["contract_sha256"] = sha256_bytes(canonical_json(value))
    return value


def submission_template(contract: Mapping[str, Any]) -> dict[str, Any]:
    role = str(contract["role"])
    is_adjudicator = role == "adjudicator"
    rows = [
        {"calibration_id": row["calibration_id"], "label": "", "notes": None}
        for row in CALIBRATION_ITEMS
    ]
    return {
        "calibration": {
            "locked": False,
            "responses": rows,
            "responses_sha256": ZERO_SHA256,
            "summary": {
                "completed_count": 0,
                "correct_count": 0,
                "qualified": False,
                "total_count": len(rows),
            },
        },
        "challenge": contract["challenge"],
        "conflict_declaration": {
            "conflict_free": False,
            "coordinator_submission": False,
            "independent_principal": False,
            "will_participate_in_adjudication": is_adjudicator,
            "will_submit_original_annotations": not is_adjudicator,
        },
        "contract_sha256": contract["contract_sha256"],
        "data_handling": {
            "no_credential_or_identity_disclosure": False,
            "no_real_package_content_in_calibration": False,
            "synthetic_calibration_only": True,
        },
        "lifecycle": "active",
        "principal_commitment": ZERO_SHA256,
        "principal_id": "prn_0000000000000000",
        "qualification_sha256": ZERO_SHA256,
        "role": role,
        "rubric_acknowledged": False,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "submitted_by_role": "principal_self",
        "submission_protocol": SUBMISSION_PROTOCOL,
        "synthetic_principal": False,
        "valid_for_source_commit": SOURCE_COMMIT,
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_kit(
    repository_root: Path,
    protocol: Mapping[str, Any],
    *,
    challenge: str,
    role: str,
    output: Path,
) -> dict[str, Any]:
    contract = build_contract(protocol, challenge=challenge, role=role)
    runtime = repository_root / "scripts/human_annotator_qualification_runtime.py"
    public_items = [
        {
            "calibration_id": row["calibration_id"],
            "paper_summary": row["paper_summary"],
            "query_summary": row["query_summary"],
            "synthetic_only": True,
        }
        for row in CALIBRATION_ITEMS
    ]
    files = {
        "README.txt": (
            "Synthetic-only offline qualification kit. It contains no real "
            "Record160/471 item, global opaque item ID, arm, strategy, gold, "
            "qrels, private mapping, or identity field. Hashes prove content "
            "integrity, not a person's identity.\n"
        ).encode(),
        "calibration_items.json": canonical_json(public_items),
        "contract.json": canonical_json(contract),
        "submission_template.json": canonical_json(submission_template(contract)),
        "verify.py": runtime.read_bytes(),
    }
    manifest: dict[str, Any] = {
        "challenge": challenge,
        "contract_sha256": contract["contract_sha256"],
        "files": [
            {"path": name, "sha256": sha256_bytes(raw), "size": len(raw)}
            for name, raw in sorted(files.items())
        ],
        "manifest_sha256": ZERO_SHA256,
        "protocol": KIT_PROTOCOL,
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    manifest["manifest_sha256"] = _digest_without(manifest, "manifest_sha256")
    files["manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(_zip_info(name), raw)
    return {
        "challenge": challenge,
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "kit_sha256": sha256_file(output),
        "protocol": PROTOCOL,
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "status": "qualification_kit_built",
    }


def _read_archive(path: Path) -> dict[str, bytes]:
    try:
        if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise HumanAnnotatorQualificationError("archive_size_or_presence_invalid")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise HumanAnnotatorQualificationError("archive_member_limit")
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
                    raise HumanAnnotatorQualificationError("archive_member_unsafe")
                files[name] = archive.read(info)
    except HumanAnnotatorQualificationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise HumanAnnotatorQualificationError("archive_invalid") from exc
    return files


def verify_kit(
    path: Path, protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    files = _read_archive(path)
    expected = {
        "README.txt",
        "calibration_items.json",
        "contract.json",
        "manifest.json",
        "submission_template.json",
        "verify.py",
    }
    if set(files) != expected:
        raise HumanAnnotatorQualificationError("kit_inventory_invalid")
    manifest = decode_object(files["manifest.json"], reason="kit_manifest_invalid")
    if set(manifest) != {
        "challenge",
        "contract_sha256",
        "files",
        "manifest_sha256",
        "protocol",
        "role",
        "schema_version",
        "source_commit",
    } or _digest_without(manifest, "manifest_sha256") != manifest["manifest_sha256"]:
        raise HumanAnnotatorQualificationError("kit_manifest_invalid")
    if (
        manifest["protocol"] != KIT_PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["role"] not in ROLES
    ):
        raise HumanAnnotatorQualificationError("kit_binding_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list) or len(inventory) != 5:
        raise HumanAnnotatorQualificationError("kit_inventory_invalid")
    seen: set[str] = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or row["path"] in seen
            or row["path"] not in files
            or row["path"] == "manifest.json"
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != sha256_bytes(files[row["path"]])
        ):
            raise HumanAnnotatorQualificationError("kit_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise HumanAnnotatorQualificationError("kit_inventory_invalid")
    contract = decode_object(files["contract.json"], reason="kit_contract_invalid")
    expected_contract = build_contract(
        protocol, challenge=manifest["challenge"], role=manifest["role"]
    )
    if contract != expected_contract:
        raise HumanAnnotatorQualificationError("kit_contract_binding_invalid")
    if files["verify.py"] != (
        repository_root / "scripts/human_annotator_qualification_runtime.py"
    ).read_bytes():
        raise HumanAnnotatorQualificationError("kit_runtime_drift")
    if decode_object(
        files["submission_template.json"], reason="kit_template_invalid"
    ) != submission_template(contract):
        raise HumanAnnotatorQualificationError("kit_template_drift")
    public_items = json.loads(files["calibration_items.json"].decode("utf-8"))
    if not isinstance(public_items, list) or len(public_items) != len(CALIBRATION_ITEMS):
        raise HumanAnnotatorQualificationError("kit_calibration_invalid")
    if _walk_keys(public_items) & {
        "arm",
        "case_id",
        "gold",
        "opaque_id",
        "private_mapping",
        "qrels",
        "strategy",
        "target_paper",
    }:
        raise HumanAnnotatorQualificationError("kit_real_evidence_leak")
    return contract


def _calibration_hash(rows: Any) -> str:
    return sha256_bytes(canonical_json(rows))


def _validate_notes(value: Any) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) > MAX_NOTES_LENGTH
        or any(ord(char) < 32 and char not in "\t\n" for char in value)
        or re.match(r"^[\s\t]*[=+\-@]", value)
    ):
        raise HumanAnnotatorQualificationError("calibration_notes_invalid")


def verify_submission(
    kit_path: Path,
    submission_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    contract = verify_kit(kit_path, protocol, repository_root=repository_root)
    submission = read_object(submission_path)
    if set(submission) != set(submission_template(contract)):
        raise HumanAnnotatorQualificationError("qualification_schema_invalid")
    if (
        submission["submission_protocol"] != SUBMISSION_PROTOCOL
        or submission["schema_version"] != SCHEMA_VERSION
        or submission["source_commit"] != SOURCE_COMMIT
        or submission["valid_for_source_commit"] != SOURCE_COMMIT
        or submission["challenge"] != contract["challenge"]
        or submission["contract_sha256"] != contract["contract_sha256"]
        or submission["role"] != contract["role"]
        or _digest_without(submission, "qualification_sha256")
        != submission["qualification_sha256"]
    ):
        raise HumanAnnotatorQualificationError("qualification_binding_invalid")
    if _walk_keys(submission) & FORBIDDEN_IDENTITY_KEYS:
        raise HumanAnnotatorQualificationError("identity_field_forbidden")
    if (
        not isinstance(submission["principal_id"], str)
        or not PRINCIPAL_RE.fullmatch(submission["principal_id"])
        or not isinstance(submission["principal_commitment"], str)
        or not SHA256_RE.fullmatch(submission["principal_commitment"])
        or submission["principal_commitment"] == ZERO_SHA256
    ):
        raise HumanAnnotatorQualificationError("opaque_principal_invalid")
    if submission["submitted_by_role"] != "principal_self":
        raise HumanAnnotatorQualificationError("coordinator_proxy_forbidden")
    if submission["lifecycle"] != "active":
        raise HumanAnnotatorQualificationError("qualification_not_active")
    if submission["synthetic_principal"] is not False and not (
        allow_synthetic and submission["synthetic_principal"] is True
    ):
        raise HumanAnnotatorQualificationError("synthetic_principal_not_real")
    role = submission["role"]
    expected_conflict = {
        "conflict_free": True,
        "coordinator_submission": False,
        "independent_principal": True,
        "will_participate_in_adjudication": role == "adjudicator",
        "will_submit_original_annotations": role != "adjudicator",
    }
    if submission["conflict_declaration"] != expected_conflict:
        raise HumanAnnotatorQualificationError("role_conflict_declaration_invalid")
    if submission["data_handling"] != {
        "no_credential_or_identity_disclosure": True,
        "no_real_package_content_in_calibration": True,
        "synthetic_calibration_only": True,
    } or submission["rubric_acknowledged"] is not True:
        raise HumanAnnotatorQualificationError("rubric_or_data_handling_invalid")
    calibration = submission["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != {
        "locked",
        "responses",
        "responses_sha256",
        "summary",
    }:
        raise HumanAnnotatorQualificationError("calibration_schema_invalid")
    rows = calibration["responses"]
    if not isinstance(rows, list) or len(rows) != len(CALIBRATION_ITEMS):
        raise HumanAnnotatorQualificationError("calibration_incomplete")
    seen: set[str] = set()
    correct = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "calibration_id",
            "label",
            "notes",
        }:
            raise HumanAnnotatorQualificationError("calibration_row_invalid")
        item_id = row["calibration_id"]
        label = row["label"]
        if item_id in seen or item_id not in CALIBRATION_REFERENCE:
            raise HumanAnnotatorQualificationError("calibration_identity_invalid")
        if label not in LABELS:
            raise HumanAnnotatorQualificationError("calibration_label_invalid")
        _validate_notes(row["notes"])
        seen.add(item_id)
        correct += int(label == CALIBRATION_REFERENCE[item_id])
    if set(CALIBRATION_REFERENCE) != seen:
        raise HumanAnnotatorQualificationError("calibration_incomplete")
    if (
        calibration["locked"] is not True
        or calibration["responses_sha256"] != _calibration_hash(rows)
        or calibration["summary"]
        != {
            "completed_count": len(rows),
            "correct_count": correct,
            "qualified": correct == len(rows),
            "total_count": len(rows),
        }
    ):
        raise HumanAnnotatorQualificationError("calibration_lock_or_summary_invalid")
    if correct != len(rows):
        raise HumanAnnotatorQualificationError("calibration_not_qualified")
    return copy.deepcopy(submission)


def build_synthetic_submission(
    contract: Mapping[str, Any],
    output: Path,
    *,
    principal_id: str,
    principal_commitment: str,
    scenario: str = "qualified_three_roles",
) -> None:
    value = submission_template(contract)
    role = str(contract["role"])
    rows = [
        {
            "calibration_id": item_id,
            "label": label,
            "notes": "synthetic calibration",
        }
        for item_id, label in CALIBRATION_REFERENCE.items()
    ]
    value.update(
        {
            "calibration": {
                "locked": True,
                "responses": rows,
                "responses_sha256": _calibration_hash(rows),
                "summary": {
                    "completed_count": len(rows),
                    "correct_count": len(rows),
                    "qualified": True,
                    "total_count": len(rows),
                },
            },
            "conflict_declaration": {
                "conflict_free": True,
                "coordinator_submission": False,
                "independent_principal": True,
                "will_participate_in_adjudication": role == "adjudicator",
                "will_submit_original_annotations": role != "adjudicator",
            },
            "data_handling": {
                "no_credential_or_identity_disclosure": True,
                "no_real_package_content_in_calibration": True,
                "synthetic_calibration_only": True,
            },
            "principal_commitment": principal_commitment,
            "principal_id": principal_id,
            "rubric_acknowledged": True,
            "synthetic_principal": True,
        }
    )
    if scenario == "coordinator_proxy":
        value["submitted_by_role"] = "human_package_coordinator"
    elif scenario == "calibration_incomplete":
        rows.pop()
        value["calibration"]["responses_sha256"] = _calibration_hash(rows)
        value["calibration"]["summary"]["completed_count"] = len(rows)
        value["calibration"]["summary"]["total_count"] = len(CALIBRATION_ITEMS)
    elif scenario == "illegal_label":
        rows[0]["label"] = "maybe"
        value["calibration"]["responses_sha256"] = _calibration_hash(rows)
    elif scenario == "commit_drift":
        value["source_commit"] = "f" * 40
    elif scenario == "revoked_reuse":
        value["lifecycle"] = "revoked"
    elif scenario == "expired_qualification":
        value["valid_for_source_commit"] = "0" * 40
    value["qualification_sha256"] = _digest_without(
        value, "qualification_sha256"
    )
    if scenario == "qualification_tamper":
        value["rubric_acknowledged"] = False
    output.write_bytes(canonical_json(value))


def _proposal(
    submissions: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    previous_sha256: str,
    sequence: int,
) -> dict[str, Any]:
    by_role = {str(row["role"]): row for row in submissions}
    if set(by_role) != set(ROLES) or len(by_role) != len(submissions):
        raise HumanAnnotatorQualificationError("role_population_invalid")
    principals = [str(by_role[role]["principal_id"]) for role in ROLES]
    commitments = [str(by_role[role]["principal_commitment"]) for role in ROLES]
    if len(set(principals)) != 3:
        raise HumanAnnotatorQualificationError("principal_role_conflict")
    if len(set(commitments)) != 3:
        raise HumanAnnotatorQualificationError("principal_alias_rebinding")
    value: dict[str, Any] = {
        "assignments": [
            {
                "principal_commitment": by_role[role]["principal_commitment"],
                "principal_id": by_role[role]["principal_id"],
                "qualification_sha256": by_role[role]["qualification_sha256"],
                "role": role,
            }
            for role in ROLES
        ],
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "previous_sha256": previous_sha256,
        "proposal_sha256": ZERO_SHA256,
        "protocol": ASSIGNMENT_PROTOCOL,
        "qualification_protocol_sha256": protocol["protocol_sha256"],
        "real_package_distributed": False,
        "schema_version": SCHEMA_VERSION,
        "separation_of_duties_sha256": protocol["bindings"][
            "separation_of_duties"
        ]["sha256"],
        "sequence": sequence,
        "source_commit": SOURCE_COMMIT,
        "status": "ready_for_real_assignment",
    }
    value["proposal_sha256"] = _digest_without(value, "proposal_sha256")
    return value


def import_dry_run(
    inputs: list[tuple[Path, Path]],
    ledger_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    proposal_path: Path | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    submissions = [
        verify_submission(
            kit,
            submission,
            protocol,
            repository_root=repository_root,
            allow_synthetic=allow_synthetic,
        )
        for kit, submission in inputs
    ]
    ledger = (
        read_object(ledger_path)
        if ledger_path.exists()
        else {"consumed": [], "proposals": [], "protocol": LEDGER_PROTOCOL}
    )
    if (
        set(ledger) != {"consumed", "proposals", "protocol"}
        or ledger["protocol"] != LEDGER_PROTOCOL
    ):
        raise HumanAnnotatorQualificationError("intake_ledger_invalid")
    if not isinstance(ledger["consumed"], list) or not isinstance(
        ledger["proposals"], list
    ):
        raise HumanAnnotatorQualificationError("intake_ledger_invalid")
    previous = ZERO_SHA256
    for expected_sequence, row in enumerate(ledger["proposals"], 1):
        if (
            not isinstance(row, dict)
            or set(row) != {"previous_sha256", "proposal_sha256", "sequence"}
            or row["sequence"] != expected_sequence
            or row["previous_sha256"] != previous
            or not isinstance(row["proposal_sha256"], str)
            or not SHA256_RE.fullmatch(row["proposal_sha256"])
        ):
            raise HumanAnnotatorQualificationError("proposal_ledger_chain_invalid")
        previous = row["proposal_sha256"]
    consumed = {
        row["challenge"]
        for row in ledger["consumed"]
        if isinstance(row, dict) and isinstance(row.get("challenge"), str)
    }
    challenges = [str(row["challenge"]) for row in submissions]
    if len(set(challenges)) != 3 or any(value in consumed for value in challenges):
        raise HumanAnnotatorQualificationError("challenge_replay")
    proposal = _proposal(
        submissions,
        protocol,
        previous_sha256=previous,
        sequence=len(ledger["proposals"]) + 1,
    )
    ledger["consumed"].extend(
        {
            "challenge": row["challenge"],
            "qualification_sha256": row["qualification_sha256"],
            "role": row["role"],
        }
        for row in submissions
    )
    ledger["consumed"] = sorted(
        ledger["consumed"], key=lambda row: (row["challenge"], row["role"])
    )
    ledger["proposals"].append(
        {
            "previous_sha256": proposal["previous_sha256"],
            "proposal_sha256": proposal["proposal_sha256"],
            "sequence": proposal["sequence"],
        }
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(canonical_json(ledger))
    if proposal_path is not None:
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.write_bytes(canonical_json(proposal))
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "protocol": PROTOCOL,
        "real_package_distributed": False,
        "role_count": 3,
        "schema_version": SCHEMA_VERSION,
        "status": "annotator_roles_qualified",
    }


def simulate_matrix(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="human-annotator-qualification-matrix-"
    ) as temp_name:
        root = Path(temp_name)
        kits: dict[str, Path] = {}
        contracts: dict[str, dict[str, Any]] = {}
        for ordinal, role in enumerate(ROLES, 1):
            kit = root / f"{role}.zip"
            challenge = sha256_bytes(f"synthetic-{role}".encode())
            build_kit(
                repository_root,
                protocol,
                challenge=challenge,
                role=role,
                output=kit,
            )
            kits[role] = kit
            contracts[role] = verify_kit(
                kit, protocol, repository_root=repository_root
            )
        for scenario in protocol["synthetic_scenarios"]["names"]:
            scenario_root = root / scenario
            scenario_root.mkdir()
            submissions: dict[str, Path] = {}
            principal_ids = {
                role: f"prn_{ordinal:016x}"
                for ordinal, role in enumerate(ROLES, 1)
            }
            commitments = {
                role: sha256_bytes(f"commitment:{role}".encode())
                for role in ROLES
            }
            if scenario == "same_principal":
                principal_ids["annotator_b"] = principal_ids["annotator_a"]
            if scenario == "annotator_is_adjudicator":
                principal_ids["adjudicator"] = principal_ids["annotator_a"]
            if scenario == "identity_alias":
                commitments["annotator_b"] = commitments["annotator_a"]
            local_scenario = scenario if scenario in {
                "coordinator_proxy",
                "calibration_incomplete",
                "illegal_label",
                "commit_drift",
                "qualification_tamper",
                "revoked_reuse",
                "expired_qualification",
            } else "qualified_three_roles"
            for role in ROLES:
                submission = scenario_root / f"{role}.json"
                build_synthetic_submission(
                    contracts[role],
                    submission,
                    principal_id=principal_ids[role],
                    principal_commitment=commitments[role],
                    scenario=local_scenario if role == "annotator_a" else "qualified_three_roles",
                )
                submissions[role] = submission
            expected = (
                "passed" if scenario == "qualified_three_roles" else "rejected"
            )
            observed = "passed"
            reason = None
            try:
                ledger = scenario_root / "ledger.json"
                inputs = [(kits[role], submissions[role]) for role in ROLES]
                import_dry_run(
                    inputs,
                    ledger,
                    protocol,
                    repository_root=repository_root,
                    allow_synthetic=True,
                )
                if scenario == "challenge_replay":
                    import_dry_run(
                        inputs,
                        ledger,
                        protocol,
                        repository_root=repository_root,
                        allow_synthetic=True,
                    )
            except HumanAnnotatorQualificationError as exc:
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
        raise HumanAnnotatorQualificationError("synthetic_matrix_mismatch")
    return {
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_QUALIFIED,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "passed_count": len(rows),
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "real_package_distributed": False,
        "scenario_count": len(rows),
        "scenarios": rows,
        "schema_version": SCHEMA_VERSION,
        "status": "annotator_roles_qualified",
    }


def audit_readiness(_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blocked_reasons": [
            "annotator_a_real_qualification_missing",
            "annotator_b_real_qualification_missing",
            "adjudicator_real_qualification_missing",
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(REAL_BLOCKERS),
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "real_package_distributed": False,
        "role_assignment_state": "not_ready_missing_real_qualified_principals",
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_real_qualified_principals",
    }
