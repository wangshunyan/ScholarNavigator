#!/usr/bin/env python3
"""Standard-library verifier for provider capacity declaration kits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


PROTOCOL = "provider_capacity_declaration_intake_v1"
DECLARATION_PROTOCOL = "provider_capacity_declaration_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
MAX_JSON_BYTES = 2 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class DeclarationError(RuntimeError):
    """Fail-closed declaration verifier error."""


class UsageError(DeclarationError):
    """Invalid command line."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def canonical_bytes(value: object) -> bytes:
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


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise DeclarationError("duplicate_json_key")
        result[key] = value
    return result


def read_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_JSON_BYTES:
            raise DeclarationError("json_size_limit")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DeclarationError("nonfinite_json_number")
            ),
        )
    except DeclarationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeclarationError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise DeclarationError("json_root_not_object")
    return value


def _exact(
    value: object, fields: set[str], reason: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DeclarationError(reason)
    return value


def _digest_declaration(value: dict[str, object]) -> str:
    payload = dict(value)
    payload["declaration_sha256"] = "0" * 64
    return stable_hash(payload)


def validate_contract(value: object) -> dict[str, object]:
    contract = _exact(
        value,
        {
            "allowed_capacity_models",
            "allowed_effective_conditions",
            "allowed_evidence_types",
            "allowed_invalidation_conditions",
            "allowed_lifecycle_states",
            "allowed_retry_after_semantics",
            "api_scope_alias",
            "bindings",
            "challenge",
            "contract_sha256",
            "formal_validation_complete",
            "privacy",
            "protocol",
            "schema_version",
            "source",
            "source_commit",
            "units",
        },
        "contract_schema_invalid",
    )
    claimed = contract["contract_sha256"]
    payload = dict(contract)
    payload.pop("contract_sha256")
    if (
        contract["protocol"] != PROTOCOL
        or contract["schema_version"] != SCHEMA_VERSION
        or contract["formal_validation_complete"] is not False
        or not isinstance(claimed, str)
        or stable_hash(payload) != claimed
    ):
        raise DeclarationError("contract_semantic_invalid")
    if (
        not isinstance(contract["source"], str)
        or not isinstance(contract["api_scope_alias"], str)
        or not isinstance(contract["source_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", contract["source_commit"])
    ):
        raise DeclarationError("contract_binding_invalid")
    challenge = _exact(
        contract["challenge"],
        {"challenge_id", "issued_epoch", "max_age_seconds", "one_time"},
        "challenge_contract_invalid",
    )
    if (
        not isinstance(challenge["challenge_id"], str)
        or not SHA256_RE.fullmatch(challenge["challenge_id"])
        or isinstance(challenge["issued_epoch"], bool)
        or not isinstance(challenge["issued_epoch"], int)
        or challenge["issued_epoch"] < 0
        or challenge["max_age_seconds"] != 604800
        or challenge["one_time"] is not True
    ):
        raise DeclarationError("challenge_contract_invalid")
    if contract["privacy"] != {
        "absolute_paths_allowed": False,
        "credentials_allowed": False,
        "endpoints_allowed": False,
        "free_text_allowed": False,
        "query_text_allowed": False,
        "request_headers_allowed": False,
        "url_parameters_allowed": False,
    }:
        raise DeclarationError("privacy_contract_invalid")
    return contract


def validate_declaration(
    contract: dict[str, object],
    value: object,
    current_epoch: int,
) -> dict[str, object]:
    declaration = _exact(
        value,
        {
            "api_scope_alias",
            "capacity_model",
            "challenge_id",
            "contract_sha256",
            "declaration_sha256",
            "declaration_version",
            "evidence_type",
            "effective_condition",
            "invalidation_condition",
            "lifecycle_status",
            "limits",
            "protocol",
            "retry_after_semantics",
            "schema_version",
            "source",
            "source_commit",
            "supersedes_declaration_sha256",
            "synthetic_only",
            "units",
            "valid_from_epoch",
            "valid_until_epoch",
        },
        "declaration_schema_invalid",
    )
    if (
        declaration["protocol"] != DECLARATION_PROTOCOL
        or declaration["schema_version"] != SCHEMA_VERSION
        or declaration["source_commit"] != contract["source_commit"]
        or declaration["source"] != contract["source"]
        or declaration["api_scope_alias"] != contract["api_scope_alias"]
        or declaration["challenge_id"] != contract["challenge"]["challenge_id"]
        or declaration["contract_sha256"] != contract["contract_sha256"]
    ):
        raise DeclarationError("declaration_binding_invalid")
    if (
        not isinstance(declaration["declaration_version"], str)
        or not VERSION_RE.fullmatch(declaration["declaration_version"])
        or declaration["evidence_type"]
        not in contract["allowed_evidence_types"]
        or declaration["capacity_model"]
        not in contract["allowed_capacity_models"]
        or declaration["effective_condition"]
        not in contract["allowed_effective_conditions"]
        or declaration["invalidation_condition"]
        not in contract["allowed_invalidation_conditions"]
        or declaration["retry_after_semantics"]
        not in contract["allowed_retry_after_semantics"]
        or declaration["lifecycle_status"] != "active"
        or declaration["units"] != contract["units"]
        or declaration["synthetic_only"] is not False
    ):
        raise DeclarationError("declaration_semantic_invalid")
    limits = _exact(
        declaration["limits"],
        {
            "burst",
            "cooldown_seconds",
            "max_concurrency",
            "requests_per_minute",
            "requests_per_second",
        },
        "capacity_limits_schema_invalid",
    )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in limits.values()
    ):
        raise DeclarationError("capacity_value_invalid")
    if (
        limits["burst"] < limits["max_concurrency"]
        or limits["requests_per_minute"] < limits["requests_per_second"]
    ):
        raise DeclarationError("capacity_window_contradiction")
    start = declaration["valid_from_epoch"]
    end = declaration["valid_until_epoch"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
        or current_epoch < start
        or current_epoch > end
        or current_epoch
        > contract["challenge"]["issued_epoch"]
        + contract["challenge"]["max_age_seconds"]
    ):
        raise DeclarationError("declaration_expired")
    supersedes = declaration["supersedes_declaration_sha256"]
    if supersedes != NOT_AVAILABLE and (
        not isinstance(supersedes, str) or not SHA256_RE.fullmatch(supersedes)
    ):
        raise DeclarationError("supersession_invalid")
    if _digest_declaration(declaration) != declaration["declaration_sha256"]:
        raise DeclarationError("declaration_digest_invalid")
    return declaration


def parser() -> argparse.ArgumentParser:
    result = Parser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--contract", required=True)
    verify.add_argument("--declaration", required=True)
    verify.add_argument("--current-epoch", required=True, type=int)
    return result


def status(value: str, code: int, reason: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": value,
        "exit_code": code,
        "network_request_count": 0,
    }
    if reason:
        result["reason_code"] = reason.split(":", 1)[0]
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        contract = validate_contract(read_object(Path(args.contract)))
        declaration = validate_declaration(
            contract,
            read_object(Path(args.declaration)),
            args.current_epoch,
        )
        report = {
            **status("capacity_declaration_verified", EXIT_READY),
            "source": declaration["source"],
            "declaration_sha256": declaration["declaration_sha256"],
        }
    except UsageError:
        report = status("usage_error", EXIT_USAGE, "invalid_arguments")
    except (
        DeclarationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = status(
            "declaration_or_import_violation",
            EXIT_VIOLATION,
            str(exc)
            if isinstance(exc, DeclarationError)
            else "controlled_schema_or_filesystem_violation",
        )
    try:
        sys.stdout.buffer.write(canonical_bytes(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
