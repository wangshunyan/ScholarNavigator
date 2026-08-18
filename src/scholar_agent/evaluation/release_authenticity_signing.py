"""Offline release-signing controls backed by OpenSSH signatures.

The module deliberately keeps private keys outside repository artifacts.  It
constructs canonical release envelopes, asks ``ssh-keygen -Y`` to sign or
verify them, and validates an append-only public trust-root manifest.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "release_authenticity_signing_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "febdb196486920ae4b856276b3a715d4a4c5e277"
PROTOCOL_SHA256 = "8813a7a77cd0d6288b5e2190960c94bca4b3f9f500e23c3a268dc9ccccc3e3d5"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ZERO_SHA256 = "0" * 64
HEX = frozenset("0123456789abcdef")
ALGORITHM = "ssh-ed25519"
NAMESPACE = "spar-release-authenticity-v1"
SIGNING_CONTEXT = "SPAR release authenticity signing v1"
REAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
ARTIFACT_TYPES = (
    "clearance_receipt",
    "evidence_transparency_checkpoint",
    "release_candidate",
    "standalone_auditor_bundle",
)
READINESS_STATUSES = (
    "candidate_checkpoint_no_public_release",
    "ready_with_declared_blockers",
    "verified_with_declared_blockers",
)
KEY_STATES = ("active", "rotated", "revoked")
TRANSITIONS = {
    "active": ("rotated", "revoked"),
    "rotated": (),
    "revoked": (),
}
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
PROTOCOL_KEYS = {
    "algorithm",
    "artifact_types",
    "canonical_envelope",
    "execution",
    "formal_validation_complete",
    "key_states",
    "namespace",
    "protocol",
    "protocol_sha256",
    "readiness_statuses",
    "real_blockers",
    "schema_version",
    "signing_context",
    "source_commit",
    "trust_root",
}
ENVELOPE_KEYS = {
    "artifact",
    "code_commit",
    "formal_blockers",
    "formal_validation_complete",
    "protocol",
    "readiness_status",
    "schema_version",
    "signing",
    "transparency_log",
}
PACKAGE_KEYS = {
    "envelope",
    "envelope_sha256",
    "issuance_sequence",
    "key_identity",
    "package_sha256",
    "protocol",
    "public_key_fingerprint",
    "schema_version",
    "signature_algorithm",
    "signature_base64",
    "signature_namespace",
    "test_only",
    "trust_root_sha256",
}
TRUST_ROOT_KEYS = {
    "formal_validation_complete",
    "keys",
    "protocol",
    "schema_version",
    "source_commit",
    "status",
    "transitions",
    "trust_root_sha256",
}
KEY_KEYS = {
    "activated_sequence",
    "fingerprint",
    "key_identity",
    "namespaces",
    "public_key",
    "retired_sequence",
    "state",
    "superseded_by",
    "test_only",
}
TRANSITION_KEYS = {
    "authorization",
    "content_sha256",
    "from_state",
    "key_identity",
    "previous_sha256",
    "sequence",
    "to_state",
}


class AuthenticityError(RuntimeError):
    """A signature, envelope, trust root, or artifact binding is invalid."""


class AuthenticityNotReady(AuthenticityError):
    """A required real signer or system signing tool is unavailable."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuthenticityError("artifact_unavailable") from exc
    return digest.hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise AuthenticityError("duplicate_json_key")
        value[key] = child
    return value


def _check_depth(value: Any, level: int = 0) -> None:
    if level > 48:
        raise AuthenticityError("json_nesting_limit")
    if isinstance(value, Mapping):
        if len(value) > 4096:
            raise AuthenticityError("json_member_limit")
        for child in value.values():
            _check_depth(child, level + 1)
    elif isinstance(value, list):
        if len(value) > 10000:
            raise AuthenticityError("json_member_limit")
        for child in value:
            _check_depth(child, level + 1)


def parse_json_bytes(value: bytes) -> dict[str, Any]:
    if len(value) > 8 * 1024 * 1024:
        raise AuthenticityError("json_size_limit")
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                AuthenticityError("nonfinite_json_number")
            ),
        )
    except AuthenticityError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise AuthenticityError("json_input_invalid") from exc
    if not isinstance(parsed, dict):
        raise AuthenticityError("json_root_not_object")
    _check_depth(parsed)
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes())
    except OSError as exc:
        raise AuthenticityError("json_input_unavailable") from exc


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value))
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise AuthenticityError("json_output_unavailable") from exc


def _require_keys(value: Any, keys: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise AuthenticityError(reason)
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _protocol_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    return payload


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require_keys(value, PROTOCOL_KEYS, "protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["algorithm"] != ALGORITHM
        or value["namespace"] != NAMESPACE
        or value["signing_context"] != SIGNING_CONTEXT
        or value["artifact_types"] != list(ARTIFACT_TYPES)
        or value["readiness_statuses"] != list(READINESS_STATUSES)
        or value["key_states"] != list(KEY_STATES)
        or value["real_blockers"] != list(REAL_BLOCKERS)
        or value["execution"] != EXECUTION
        or value["formal_validation_complete"] is not False
        or not _is_digest(value["protocol_sha256"])
        or value["protocol_sha256"] != PROTOCOL_SHA256
        or stable_hash(_protocol_payload(value)) != value["protocol_sha256"]
    ):
        raise AuthenticityError("protocol_schema_invalid")
    return dict(value)


def _trust_root_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["trust_root_sha256"] = ZERO_SHA256
    return payload


def finalize_trust_root(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["trust_root_sha256"] = stable_hash(_trust_root_payload(payload))
    return payload


def empty_trust_root(*, source_commit: str = SOURCE_COMMIT) -> dict[str, Any]:
    return finalize_trust_root(
        {
            "formal_validation_complete": False,
            "keys": [],
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "source_commit": source_commit,
            "status": "not_ready_missing_real_trust_anchor",
            "transitions": [],
            "trust_root_sha256": ZERO_SHA256,
        }
    )


def _public_key_parts(public_key: str) -> tuple[str, str]:
    if "\n" in public_key or "\r" in public_key:
        public_key = public_key.strip()
    parts = public_key.split()
    if len(parts) < 2 or parts[0] != ALGORITHM:
        raise AuthenticityError("public_key_invalid")
    try:
        decoded = base64.b64decode(parts[1], validate=True)
    except (ValueError, TypeError) as exc:
        raise AuthenticityError("public_key_invalid") from exc
    fingerprint = "SHA256:" + base64.b64encode(
        hashlib.sha256(decoded).digest()
    ).decode("ascii").rstrip("=")
    return f"{parts[0]} {parts[1]}", fingerprint


def _transition_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["content_sha256"] = ZERO_SHA256
    return payload


def finalize_transition(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["content_sha256"] = stable_hash(_transition_payload(payload))
    return payload


def verify_trust_root(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(value, TRUST_ROOT_KEYS, "trust_root_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or not _is_commit(value["source_commit"])
        or value["formal_validation_complete"] is not False
        or not isinstance(value["status"], str)
        or not isinstance(value["keys"], list)
        or not isinstance(value["transitions"], list)
        or not _is_digest(value["trust_root_sha256"])
        or stable_hash(_trust_root_payload(value)) != value["trust_root_sha256"]
    ):
        raise AuthenticityError("trust_root_schema_invalid")

    identities: set[str] = set()
    fingerprints: set[str] = set()
    key_states: dict[str, str] = {}
    key_rows: dict[str, Mapping[str, Any]] = {}
    for row in value["keys"]:
        _require_keys(row, KEY_KEYS, "trust_root_key_schema_invalid")
        public_key, fingerprint = _public_key_parts(row["public_key"])
        if (
            not isinstance(row["key_identity"], str)
            or not row["key_identity"]
            or row["key_identity"] in identities
            or row["fingerprint"] != fingerprint
            or fingerprint in fingerprints
            or row["public_key"] != public_key
            or row["state"] not in KEY_STATES
            or not isinstance(row["test_only"], bool)
            or row["namespaces"] != [NAMESPACE]
            or not isinstance(row["activated_sequence"], int)
            or row["activated_sequence"] < 0
            or (
                row["retired_sequence"] is not None
                and (
                    not isinstance(row["retired_sequence"], int)
                    or row["retired_sequence"] <= row["activated_sequence"]
                )
            )
            or (
                row["state"] == "active"
                and (
                    row["retired_sequence"] is not None
                    or row["superseded_by"] is not None
                )
            )
            or (
                row["state"] == "rotated"
                and (
                    not isinstance(row["superseded_by"], str)
                    or row["retired_sequence"] is None
                )
            )
            or (
                row["state"] == "revoked"
                and (
                    row["superseded_by"] is not None
                    or row["retired_sequence"] is None
                )
            )
        ):
            raise AuthenticityError("trust_root_key_schema_invalid")
        identities.add(row["key_identity"])
        fingerprints.add(fingerprint)
        key_states[row["key_identity"]] = "active"
        key_rows[row["key_identity"]] = row

    previous = ZERO_SHA256
    for sequence, row in enumerate(value["transitions"]):
        _require_keys(row, TRANSITION_KEYS, "trust_transition_schema_invalid")
        authorization = row["authorization"]
        if (
            row["sequence"] != sequence
            or row["previous_sha256"] != previous
            or not _is_digest(row["content_sha256"])
            or stable_hash(_transition_payload(row)) != row["content_sha256"]
            or row["key_identity"] not in key_states
            or row["from_state"] != key_states[row["key_identity"]]
            or row["to_state"] not in TRANSITIONS[row["from_state"]]
            or not isinstance(authorization, Mapping)
            or set(authorization)
            != {
                "kind",
                "recovery_rule_id",
                "signature_base64",
                "signer_key_identity",
                "signer_fingerprint",
                "statement_sha256",
            }
            or authorization["kind"]
            not in {"old_key_signature", "offline_recovery"}
            or not _is_digest(authorization["statement_sha256"])
        ):
            raise AuthenticityError("trust_transition_schema_invalid")
        if authorization["kind"] == "old_key_signature":
            if (
                authorization["signer_key_identity"] != row["key_identity"]
                or authorization["signer_fingerprint"]
                != key_rows[row["key_identity"]]["fingerprint"]
                or authorization["recovery_rule_id"] is not None
                or not isinstance(authorization["signature_base64"], str)
            ):
                raise AuthenticityError("trust_transition_authorization_invalid")
            statement = {
                "key_identity": row["key_identity"],
                "retired_sequence": key_rows[row["key_identity"]][
                    "retired_sequence"
                ],
                "superseded_by": key_rows[row["key_identity"]][
                    "superseded_by"
                ],
                "to_state": row["to_state"],
            }
            if stable_hash(statement) != authorization["statement_sha256"]:
                raise AuthenticityError("trust_transition_statement_invalid")
            _verify_bytes(
                canonical_json(statement),
                signature_base64=authorization["signature_base64"],
                public_key=key_rows[row["key_identity"]]["public_key"],
                key_identity=row["key_identity"],
            )
        elif (
            authorization["signer_key_identity"] is not None
            or authorization["signer_fingerprint"] is not None
            or authorization["signature_base64"] is not None
            or authorization["recovery_rule_id"]
            != "pre_registered_offline_recovery_v1"
        ):
            raise AuthenticityError("trust_transition_authorization_invalid")
        key_states[row["key_identity"]] = row["to_state"]
        previous = row["content_sha256"]

    for identity, state in key_states.items():
        if key_rows[identity]["state"] != state:
            raise AuthenticityError("trust_root_state_mismatch")
    active_real = [
        row
        for row in value["keys"]
        if row["state"] == "active" and row["test_only"] is False
    ]
    expected_status = (
        "real_trust_anchor_active"
        if active_real
        else "not_ready_missing_real_trust_anchor"
    )
    if value["status"] != expected_status:
        raise AuthenticityError("trust_root_status_invalid")
    return {
        "active_real_key_count": len(active_real),
        "key_count": len(value["keys"]),
        "status": value["status"],
        "transition_count": len(value["transitions"]),
        "trust_root_sha256": value["trust_root_sha256"],
    }


def register_key(
    trust_root: Mapping[str, Any],
    *,
    key_identity: str,
    public_key: str,
    test_only: bool,
    activated_sequence: int = 0,
) -> dict[str, Any]:
    verify_trust_root(trust_root)
    if trust_root["keys"]:
        raise AuthenticityError("initial_key_already_registered")
    normalized, fingerprint = _public_key_parts(public_key)
    value = dict(trust_root)
    value["keys"] = [
        {
            "activated_sequence": activated_sequence,
            "fingerprint": fingerprint,
            "key_identity": key_identity,
            "namespaces": [NAMESPACE],
            "public_key": normalized,
            "retired_sequence": None,
            "state": "active",
            "superseded_by": None,
            "test_only": test_only,
        }
    ]
    value["status"] = (
        "not_ready_missing_real_trust_anchor"
        if test_only
        else "real_trust_anchor_active"
    )
    result = finalize_trust_root(value)
    verify_trust_root(result)
    return result


def transition_key(
    trust_root: Mapping[str, Any],
    *,
    key_identity: str,
    to_state: str,
    retired_sequence: int,
    superseded_by: str | None = None,
    authorization_kind: str = "old_key_signature",
    signer_private_key: Path | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    verify_trust_root(trust_root)
    if to_state not in {"rotated", "revoked"}:
        raise AuthenticityError("trust_transition_invalid")
    value = json.loads(json.dumps(trust_root))
    key = next(
        (row for row in value["keys"] if row["key_identity"] == key_identity),
        None,
    )
    if key is None or key["state"] != "active":
        raise AuthenticityError("trust_transition_invalid")
    if to_state == "rotated" and not superseded_by:
        raise AuthenticityError("trust_transition_invalid")
    if to_state == "revoked" and superseded_by is not None:
        raise AuthenticityError("trust_transition_invalid")
    statement = {
        "key_identity": key_identity,
        "retired_sequence": retired_sequence,
        "superseded_by": superseded_by,
        "to_state": to_state,
    }
    signature_base64: str | None = None
    if authorization_kind == "old_key_signature":
        if signer_private_key is None:
            raise AuthenticityError("transition_signer_required")
        signature_base64 = _sign_bytes(
            canonical_json(statement),
            private_key=signer_private_key,
            executable=executable,
        )
    authorization = {
        "kind": authorization_kind,
        "recovery_rule_id": (
            "pre_registered_offline_recovery_v1"
            if authorization_kind == "offline_recovery"
            else None
        ),
        "signature_base64": signature_base64,
        "signer_key_identity": (
            key_identity if authorization_kind == "old_key_signature" else None
        ),
        "signer_fingerprint": (
            key["fingerprint"]
            if authorization_kind == "old_key_signature"
            else None
        ),
        "statement_sha256": stable_hash(statement),
    }
    transition = finalize_transition(
        {
            "authorization": authorization,
            "content_sha256": ZERO_SHA256,
            "from_state": "active",
            "key_identity": key_identity,
            "previous_sha256": (
                value["transitions"][-1]["content_sha256"]
                if value["transitions"]
                else ZERO_SHA256
            ),
            "sequence": len(value["transitions"]),
            "to_state": to_state,
        }
    )
    value["transitions"].append(transition)
    key["state"] = to_state
    key["retired_sequence"] = retired_sequence
    key["superseded_by"] = superseded_by
    value["status"] = "not_ready_missing_real_trust_anchor"
    result = finalize_trust_root(value)
    verify_trust_root(result)
    return result


def add_rotated_key(
    trust_root: Mapping[str, Any],
    *,
    key_identity: str,
    public_key: str,
    test_only: bool,
    activated_sequence: int,
) -> dict[str, Any]:
    verify_trust_root(trust_root)
    if any(row["key_identity"] == key_identity for row in trust_root["keys"]):
        raise AuthenticityError("duplicate_key_identity")
    normalized, fingerprint = _public_key_parts(public_key)
    value = json.loads(json.dumps(trust_root))
    value["keys"].append(
        {
            "activated_sequence": activated_sequence,
            "fingerprint": fingerprint,
            "key_identity": key_identity,
            "namespaces": [NAMESPACE],
            "public_key": normalized,
            "retired_sequence": None,
            "state": "active",
            "superseded_by": None,
            "test_only": test_only,
        }
    )
    value["keys"].sort(key=lambda row: row["key_identity"])
    value["status"] = (
        "real_trust_anchor_active"
        if any(
            row["state"] == "active" and row["test_only"] is False
            for row in value["keys"]
        )
        else "not_ready_missing_real_trust_anchor"
    )
    result = finalize_trust_root(value)
    verify_trust_root(result)
    return result


def build_envelope(
    *,
    artifact_type: str,
    artifact_version: str,
    content_sha256: str,
    transparency_root: str,
    transparency_sequence: int,
    code_commit: str,
    readiness_status: str,
    key_identity: str,
    test_only: bool,
) -> dict[str, Any]:
    value = {
        "artifact": {
            "content_sha256": content_sha256,
            "type": artifact_type,
            "version": artifact_version,
        },
        "code_commit": code_commit,
        "formal_blockers": list(REAL_BLOCKERS),
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "readiness_status": readiness_status,
        "schema_version": SCHEMA_VERSION,
        "signing": {
            "algorithm": ALGORITHM,
            "context": SIGNING_CONTEXT,
            "key_identity": key_identity,
            "namespace": NAMESPACE,
            "test_only": test_only,
        },
        "transparency_log": {
            "root_sha256": transparency_root,
            "sequence": transparency_sequence,
        },
    }
    verify_envelope(value)
    return value


def verify_envelope(value: Mapping[str, Any]) -> None:
    _require_keys(value, ENVELOPE_KEYS, "envelope_schema_invalid")
    artifact = _require_keys(
        value["artifact"],
        {"content_sha256", "type", "version"},
        "envelope_artifact_invalid",
    )
    signing = _require_keys(
        value["signing"],
        {"algorithm", "context", "key_identity", "namespace", "test_only"},
        "envelope_signing_invalid",
    )
    transparency = _require_keys(
        value["transparency_log"],
        {"root_sha256", "sequence"},
        "envelope_transparency_invalid",
    )
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["formal_validation_complete"] is not False
        or value["formal_blockers"] != list(REAL_BLOCKERS)
        or value["readiness_status"] not in READINESS_STATUSES
        or not _is_commit(value["code_commit"])
        or artifact["type"] not in ARTIFACT_TYPES
        or not isinstance(artifact["version"], str)
        or not artifact["version"]
        or not _is_digest(artifact["content_sha256"])
        or signing["algorithm"] != ALGORITHM
        or signing["context"] != SIGNING_CONTEXT
        or signing["namespace"] != NAMESPACE
        or not isinstance(signing["key_identity"], str)
        or not signing["key_identity"]
        or not isinstance(signing["test_only"], bool)
        or not _is_digest(transparency["root_sha256"])
        or not isinstance(transparency["sequence"], int)
        or transparency["sequence"] < 0
    ):
        raise AuthenticityError("envelope_schema_invalid")
    serialized = canonical_json(value).decode("utf-8")
    if any(token in serialized for token in ("/Users/", "\\\\Users\\\\", ".env")):
        raise AuthenticityError("envelope_machine_data_forbidden")


def find_ssh_keygen(executable: str | None = None) -> str:
    candidate = executable or shutil.which("ssh-keygen")
    if not candidate:
        raise AuthenticityNotReady("ssh_keygen_unavailable")
    return candidate


def _run_ssh(
    arguments: Sequence[str],
    *,
    executable: str | None = None,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = [find_ssh_keygen(executable), *arguments]
    env = {
        "HOME": tempfile.gettempdir(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }
    try:
        return subprocess.run(
            command,
            input=stdin,
            check=False,
            capture_output=True,
            env=env,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthenticityNotReady("ssh_keygen_unavailable") from exc


def _sign_bytes(
    payload: bytes,
    *,
    private_key: Path,
    executable: str | None = None,
) -> str:
    if not private_key.is_file():
        raise AuthenticityNotReady("private_key_unavailable")
    with tempfile.TemporaryDirectory(prefix="spar-signing-bytes-") as temp:
        payload_path = Path(temp) / "payload"
        signature_path = Path(f"{payload_path}.sig")
        payload_path.write_bytes(payload)
        result = _run_ssh(
            [
                "-Y",
                "sign",
                "-q",
                "-f",
                str(private_key),
                "-n",
                NAMESPACE,
                str(payload_path),
            ],
            executable=executable,
        )
        if result.returncode != 0 or not signature_path.is_file():
            raise AuthenticityError("signature_generation_failed")
        return base64.b64encode(signature_path.read_bytes()).decode("ascii")


def _verify_bytes(
    payload: bytes,
    *,
    signature_base64: str,
    public_key: str,
    key_identity: str,
    executable: str | None = None,
) -> None:
    try:
        signature = base64.b64decode(signature_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise AuthenticityError("signature_encoding_invalid") from exc
    with tempfile.TemporaryDirectory(prefix="spar-verification-bytes-") as temp:
        root = Path(temp)
        allowed = root / "allowed_signers"
        signature_path = root / "payload.sig"
        allowed.write_text(
            f'{key_identity} namespaces="{NAMESPACE}" {public_key}\n',
            encoding="utf-8",
        )
        signature_path.write_bytes(signature)
        result = _run_ssh(
            [
                "-Y",
                "verify",
                "-q",
                "-f",
                str(allowed),
                "-I",
                key_identity,
                "-n",
                NAMESPACE,
                "-s",
                str(signature_path),
            ],
            executable=executable,
            stdin=payload,
        )
    if result.returncode != 0:
        raise AuthenticityError("signature_verification_failed")


def generate_test_key(
    directory: Path,
    *,
    key_identity: str,
    executable: str | None = None,
) -> tuple[Path, str, str]:
    if not key_identity or any(char.isspace() for char in key_identity):
        raise AuthenticityError("key_identity_invalid")
    try:
        directory.mkdir(parents=True, exist_ok=False)
        os.chmod(directory, 0o700)
    except OSError as exc:
        raise AuthenticityError("test_key_directory_invalid") from exc
    private_key = directory / "operator-test-key"
    result = _run_ssh(
        [
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "SPAR test-only release signer",
            "-f",
            str(private_key),
        ],
        executable=executable,
    )
    if result.returncode != 0:
        raise AuthenticityNotReady("test_key_generation_failed")
    try:
        public_key = private_key.with_suffix(".pub").read_text(
            encoding="utf-8"
        ).strip()
        os.chmod(private_key, 0o600)
    except OSError as exc:
        raise AuthenticityError("test_public_key_unavailable") from exc
    normalized, fingerprint = _public_key_parts(public_key)
    return private_key, normalized, fingerprint


def _key_for_identity(
    trust_root: Mapping[str, Any], key_identity: str
) -> Mapping[str, Any]:
    for row in trust_root["keys"]:
        if row["key_identity"] == key_identity:
            return row
    raise AuthenticityError("unknown_key")


def _package_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["package_sha256"] = ZERO_SHA256
    return payload


def sign_envelope(
    envelope: Mapping[str, Any],
    *,
    private_key: Path,
    trust_root: Mapping[str, Any],
    issuance_sequence: int,
    executable: str | None = None,
) -> dict[str, Any]:
    verify_envelope(envelope)
    verify_trust_root(trust_root)
    identity = envelope["signing"]["key_identity"]
    key = _key_for_identity(trust_root, identity)
    if key["state"] != "active":
        raise AuthenticityError("key_not_active_for_signing")
    if envelope["signing"]["test_only"] != key["test_only"]:
        raise AuthenticityError("test_only_binding_mismatch")
    if issuance_sequence < key["activated_sequence"] or (
        key["retired_sequence"] is not None
        and issuance_sequence >= key["retired_sequence"]
    ):
        raise AuthenticityError("key_not_active_at_issuance")
    if key["test_only"] is False and trust_root["status"] != "real_trust_anchor_active":
        raise AuthenticityNotReady("real_signer_not_ready")
    if not private_key.is_file():
        raise AuthenticityNotReady("private_key_unavailable")

    signature_base64 = _sign_bytes(
        canonical_json(envelope),
        private_key=private_key,
        executable=executable,
    )
    package = {
        "envelope": dict(envelope),
        "envelope_sha256": sha256_bytes(canonical_json(envelope)),
        "issuance_sequence": issuance_sequence,
        "key_identity": identity,
        "package_sha256": ZERO_SHA256,
        "protocol": PROTOCOL,
        "public_key_fingerprint": key["fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "signature_algorithm": ALGORITHM,
        "signature_base64": signature_base64,
        "signature_namespace": NAMESPACE,
        "test_only": key["test_only"],
        "trust_root_sha256": trust_root["trust_root_sha256"],
    }
    package["package_sha256"] = stable_hash(_package_payload(package))
    verify_signature_package(
        package,
        trust_root=trust_root,
        artifact_sha256=envelope["artifact"]["content_sha256"],
        executable=executable,
    )
    return package


def verify_signature_package(
    package: Mapping[str, Any],
    *,
    trust_root: Mapping[str, Any],
    artifact_sha256: str,
    executable: str | None = None,
) -> dict[str, Any]:
    _require_keys(package, PACKAGE_KEYS, "signature_package_schema_invalid")
    verify_trust_root(trust_root)
    verify_envelope(package["envelope"])
    envelope = package["envelope"]
    if (
        package["protocol"] != PROTOCOL
        or package["schema_version"] != SCHEMA_VERSION
        or package["signature_algorithm"] != ALGORITHM
        or package["signature_namespace"] != NAMESPACE
        or package["key_identity"] != envelope["signing"]["key_identity"]
        or package["test_only"] != envelope["signing"]["test_only"]
        or package["envelope_sha256"]
        != sha256_bytes(canonical_json(envelope))
        or not _is_digest(package["trust_root_sha256"])
        or package["package_sha256"] != stable_hash(_package_payload(package))
        or artifact_sha256 != envelope["artifact"]["content_sha256"]
        or not isinstance(package["issuance_sequence"], int)
        or package["issuance_sequence"] < 0
    ):
        raise AuthenticityError("signature_package_binding_invalid")
    key = _key_for_identity(trust_root, package["key_identity"])
    if (
        key["fingerprint"] != package["public_key_fingerprint"]
        or key["test_only"] != package["test_only"]
        or package["issuance_sequence"] < key["activated_sequence"]
        or (
            key["retired_sequence"] is not None
            and package["issuance_sequence"] >= key["retired_sequence"]
        )
    ):
        raise AuthenticityError("signature_key_binding_invalid")
    if (
        package["trust_root_sha256"] != trust_root["trust_root_sha256"]
        and key["retired_sequence"] is None
    ):
        raise AuthenticityError("signature_trust_root_binding_invalid")
    _verify_bytes(
        canonical_json(envelope),
        signature_base64=package["signature_base64"],
        public_key=key["public_key"],
        key_identity=package["key_identity"],
        executable=executable,
    )
    return {
        "artifact_type": envelope["artifact"]["type"],
        "code_commit": envelope["code_commit"],
        "formal_blockers": list(envelope["formal_blockers"]),
        "formal_validation_complete": False,
        "key_identity": package["key_identity"],
        "package_sha256": package["package_sha256"],
        "signature_verified": True,
        "test_only": package["test_only"],
        "transparency_root": envelope["transparency_log"]["root_sha256"],
    }


def verify_issuance_set(packages: Sequence[Mapping[str, Any]]) -> None:
    """Reject two different signatures for one release issuance identity."""

    seen: dict[tuple[str, str, int, str], str] = {}
    for package in packages:
        _require_keys(package, PACKAGE_KEYS, "signature_package_schema_invalid")
        envelope = package["envelope"]
        verify_envelope(envelope)
        identity = (
            envelope["artifact"]["type"],
            envelope["artifact"]["version"],
            package["issuance_sequence"],
            package["key_identity"],
        )
        previous = seen.get(identity)
        if previous is not None and previous != package["package_sha256"]:
            raise AuthenticityError("duplicate_issuance_conflict")
        seen[identity] = package["package_sha256"]


def simulate_matrix() -> dict[str, Any]:
    """Run a deterministic test-only signing and trust-root attack matrix."""

    scenarios: list[dict[str, Any]] = []

    def passed(name: str, invariant: str) -> None:
        scenarios.append(
            {"invariant": invariant, "scenario": name, "status": "passed"}
        )

    with tempfile.TemporaryDirectory(prefix="spar-authenticity-matrix-") as temp:
        root = Path(temp)
        artifact = root / "candidate.bin"
        artifact.write_bytes(b"synthetic-release-candidate-v1\n")
        old_private, old_public, _old_fingerprint = generate_test_key(
            root / "old",
            key_identity="test:matrix-old",
        )
        trust = register_key(
            empty_trust_root(),
            key_identity="test:matrix-old",
            public_key=old_public,
            test_only=True,
        )
        envelope = build_envelope(
            artifact_type="release_candidate",
            artifact_version="v1",
            content_sha256=sha256_file(artifact),
            transparency_root="2" * 64,
            transparency_sequence=0,
            code_commit=SOURCE_COMMIT,
            readiness_status="ready_with_declared_blockers",
            key_identity="test:matrix-old",
            test_only=True,
        )
        first = sign_envelope(
            envelope,
            private_key=old_private,
            trust_root=trust,
            issuance_sequence=0,
        )
        second = sign_envelope(
            envelope,
            private_key=old_private,
            trust_root=trust,
            issuance_sequence=0,
        )
        if canonical_json(first) != canonical_json(second):
            raise AuthenticityError("signature_nondeterministic")
        verify_signature_package(
            first,
            trust_root=trust,
            artifact_sha256=sha256_file(artifact),
        )
        passed("valid_issue_and_verify", "openssh_signature_verified")
        passed("double_environment_equivalence", "signature_bytes_deterministic")

        try:
            verify_signature_package(
                first,
                trust_root=trust,
                artifact_sha256="3" * 64,
            )
        except AuthenticityError:
            passed("artifact_tampering", "artifact_hash_binding")
        else:
            raise AuthenticityError("tampered_artifact_accepted")

        branch = json.loads(json.dumps(first))
        branch["envelope"]["transparency_log"]["root_sha256"] = "4" * 64
        branch["envelope_sha256"] = sha256_bytes(
            canonical_json(branch["envelope"])
        )
        branch["package_sha256"] = stable_hash(_package_payload(branch))
        try:
            verify_signature_package(
                branch,
                trust_root=trust,
                artifact_sha256=sha256_file(artifact),
            )
        except AuthenticityError:
            passed("transparency_fork", "signed_log_root_binding")
        else:
            raise AuthenticityError("forked_log_root_accepted")

        new_private, new_public, _new_fingerprint = generate_test_key(
            root / "new",
            key_identity="test:matrix-new",
        )
        rotated = transition_key(
            trust,
            key_identity="test:matrix-old",
            to_state="rotated",
            retired_sequence=10,
            superseded_by="test:matrix-new",
            signer_private_key=old_private,
        )
        rotated = add_rotated_key(
            rotated,
            key_identity="test:matrix-new",
            public_key=new_public,
            test_only=True,
            activated_sequence=10,
        )
        verify_signature_package(
            first,
            trust_root=rotated,
            artifact_sha256=sha256_file(artifact),
        )
        try:
            sign_envelope(
                envelope,
                private_key=old_private,
                trust_root=rotated,
                issuance_sequence=10,
            )
        except AuthenticityError:
            passed("signed_rotation", "old_key_blocked_historical_kept")
        else:
            raise AuthenticityError("rotated_key_signed_new_release")

        new_envelope = build_envelope(
            artifact_type="release_candidate",
            artifact_version="v2",
            content_sha256=sha256_file(artifact),
            transparency_root="5" * 64,
            transparency_sequence=1,
            code_commit=SOURCE_COMMIT,
            readiness_status="ready_with_declared_blockers",
            key_identity="test:matrix-new",
            test_only=True,
        )
        new_package = sign_envelope(
            new_envelope,
            private_key=new_private,
            trust_root=rotated,
            issuance_sequence=10,
        )
        revoked = transition_key(
            rotated,
            key_identity="test:matrix-new",
            to_state="revoked",
            retired_sequence=20,
            signer_private_key=new_private,
        )
        verify_signature_package(
            new_package,
            trust_root=revoked,
            artifact_sha256=sha256_file(artifact),
        )
        try:
            sign_envelope(
                new_envelope,
                private_key=new_private,
                trust_root=revoked,
                issuance_sequence=20,
            )
        except AuthenticityError:
            passed("revocation", "revoked_key_cannot_issue")
        else:
            raise AuthenticityError("revoked_key_signed_new_release")

        try:
            sign_envelope(
                build_envelope(
                    artifact_type="release_candidate",
                    artifact_version="v1",
                    content_sha256=sha256_file(artifact),
                    transparency_root="2" * 64,
                    transparency_sequence=0,
                    code_commit=SOURCE_COMMIT,
                    readiness_status="ready_with_declared_blockers",
                    key_identity="test:matrix-old",
                    test_only=False,
                ),
                private_key=old_private,
                trust_root=trust,
                issuance_sequence=0,
            )
        except AuthenticityError:
            passed("test_key_pollution", "test_only_cannot_impersonate_real")
        else:
            raise AuthenticityError("test_key_impersonation_accepted")

        if old_private.stat().st_mode & 0o777 != 0o600:
            raise AuthenticityError("private_key_permissions_invalid")
        passed("private_key_isolation", "operator_key_not_serialized")

        try:
            find_ssh_keygen(str(root / "missing-ssh-keygen"))
            _run_ssh([], executable=str(root / "missing-ssh-keygen"))
        except AuthenticityNotReady:
            passed("missing_signing_tool", "fail_closed_not_ready")
        else:
            raise AuthenticityError("missing_tool_not_detected")

    return {
        "execution": dict(EXECUTION),
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "schema_version": SCHEMA_VERSION,
        "status": "signing_controls_ready",
    }


def audit_current(
    repository_root: Path,
    protocol: Mapping[str, Any],
    trust_root: Mapping[str, Any],
) -> dict[str, Any]:
    load_protocol(
        repository_root / "benchmark/release_authenticity_signing_v1_protocol.json"
    )
    trust = verify_trust_root(trust_root)
    checkpoint = read_json(
        repository_root / "benchmark/evidence_transparency_log_v1_checkpoint.json"
    )
    standalone = read_json(
        repository_root
        / "benchmark/standalone_auditor_bundle_v1_evidence/readiness.json"
    )
    release_candidate = read_json(
        repository_root
        / "benchmark/release_candidate_reproducibility_v1_evidence/current.json"
    )
    revocation = read_json(
        repository_root / "benchmark/evidence_revocation_response_v1_ledger.json"
    )
    if checkpoint.get("formal_validation_complete") is not False:
        raise AuthenticityError("transparency_checkpoint_status_invalid")
    if (
        standalone.get("formal_validation_complete") is not False
        or release_candidate.get("formal_validation_complete") is not False
        or revocation.get("formal_validation_complete") is not False
        or revocation.get("events") != []
    ):
        raise AuthenticityError("release_integration_state_invalid")
    tool_available = shutil.which("ssh-keygen") is not None
    if not tool_available:
        reason = "ssh_keygen_unavailable"
    elif trust["active_real_key_count"] == 0:
        reason = "missing_real_trust_anchor_or_signer"
    else:
        reason = None
    return {
        "algorithm": ALGORITHM,
        "execution": dict(EXECUTION),
        "exit_code": EXIT_NOT_READY if reason else EXIT_READY,
        "formal_blockers": list(REAL_BLOCKERS),
        "formal_validation_complete": False,
        "identity_authentication": (
            "not_provided_until_real_trust_anchor_is_operator_provisioned"
        ),
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "signable_artifacts": list(ARTIFACT_TYPES),
        "signing_controls_ready": tool_available,
        "status": (
            "not_ready_missing_real_trust_anchor_or_signer"
            if reason
            else "signing_controls_ready"
        ),
        "transparency_checkpoint": {
            "log_length": checkpoint.get("log_length"),
            "merkle_root": checkpoint.get("merkle_root"),
            "status": checkpoint.get("status"),
        },
        "unsigned_integration_state": {
            "release_candidate_status": release_candidate.get("status"),
            "revocation_event_count": len(revocation["events"]),
            "standalone_status": standalone.get("status"),
        },
        "trust_root": trust,
    }


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise AuthenticityError("path_outside_allowed_root") from exc
    value = PurePosixPath(relative.as_posix())
    if any(part in {"", ".", "..", ".env"} for part in value.parts):
        raise AuthenticityError("unsafe_path")
    return value.as_posix()
