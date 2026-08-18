"""Offline-verifiable raw provider response provenance.

``provider_ingest_provenance_v1`` is an optional observation layer for new
networked runs.  It stores the exact response bytes in a deterministic archive
and records only opaque request identities and content metadata in JSON.  The
gate replays those bytes through the production connector parsers; historical
Snapshot/Record files are never treated as parser-pre-response evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.connectors.arxiv import ATOM_NS, _parse_entry
from scholar_agent.connectors.openalex import _parse_work
from scholar_agent.connectors.pubmed import _normalize_pmid, _parse_article
from scholar_agent.connectors.semantic_scholar import _parse_paper
from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.resource_accounting import (
    ResourceLedgerV1,
    validate_resource_ledger,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


PROVENANCE_CONTRACT = "provider_ingest_provenance_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "provider_ingest_provenance_gate"
EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
RAW_ARCHIVE_NAME = "provider_ingest_raw.tar"
BUNDLE_NAME = "provider_ingest_provenance.json"
PARSER_VERSION = "production_connector_parser_v1"
MAX_RAW_MEMBERS = 10000
MAX_RAW_MEMBER_BYTES = 32 * 1024 * 1024
MAX_RAW_TOTAL_BYTES = 512 * 1024 * 1024

OpaqueIdentity = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SourceName = Literal["arxiv", "openalex", "pubmed", "semantic_scholar"]
ParserName = Literal[
    "arxiv_atom_search",
    "openalex_search",
    "pubmed_esearch",
    "pubmed_efetch",
    "semantic_scholar_search",
]
TerminalState = Literal[
    "adapter_exception",
    "capture_size_exceeded",
    "connection_failure",
    "http_error",
    "malformed_response",
    "partial_success",
    "rate_limited",
    "success",
    "timeout",
]
ReasonCode = Literal[
    "identity_invalid",
    "malformed_document",
    "malformed_record",
    "missing_required_field",
    "record_not_object",
    "unknown_schema",
]


class ProviderIngestError(RuntimeError):
    """Provenance input violates the frozen capture/replay contract."""


class ProviderIngestNotEligible(ProviderIngestError):
    """The requested historical artifact has no parser-pre raw bytes."""


class RawResponseRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    member: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_member(self) -> "RawResponseRef":
        expected = f"raw/{self.sha256}.bin"
        if self.member != expected:
            raise ValueError("raw response member is not content addressed")
        return self


class EncodingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["known", "not_available"]
    value: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "EncodingMetadata":
        if self.state == "known":
            if not self.value or self.reason is not None:
                raise ValueError("known encoding requires value only")
        elif self.value is not None or not self.reason:
            raise ValueError("unavailable encoding requires reason only")
        return self


class RejectedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_index: int | None = Field(default=None, ge=0)
    reason_code: ReasonCode
    evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ProviderAttemptEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["provider_ingest_provenance_v1"] = PROVENANCE_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    envelope_identity: OpaqueIdentity
    run_identity: OpaqueIdentity
    query_identity: OpaqueIdentity
    source: SourceName
    attempt_identity: OpaqueIdentity
    request_sequence: int = Field(ge=0)
    resource_operation_identity: OpaqueIdentity
    checkpoint_generation: int = Field(ge=0)
    manifest_identity: OpaqueIdentity
    http_status: int | None = Field(default=None, ge=100, le=599)
    content_type: str | None = None
    encoding: EncodingMetadata
    compression: Literal["identity", "gzip", "not_available"]
    raw_response: RawResponseRef | None = None
    page_index: int = Field(ge=0)
    request_cursor_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_envelope_identity: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    next_cursor_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parser_name: ParserName
    parser_version: Literal["production_connector_parser_v1"] = PARSER_VERSION
    parsed_record_count: int = Field(ge=0)
    accepted_record_count: int = Field(ge=0)
    rejected_record_count: int = Field(ge=0)
    rejections: list[RejectedRecord] = Field(default_factory=list)
    parsed_output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    terminal_state: TerminalState
    terminal_reason_code: str | None = None

    @model_validator(mode="after")
    def validate_conservation(self) -> "ProviderAttemptEnvelope":
        if self.parsed_record_count != (
            self.accepted_record_count + self.rejected_record_count
        ):
            raise ValueError("parsed/accepted/rejected count is not conserved")
        if self.rejected_record_count != len(self.rejections):
            raise ValueError("rejection detail count mismatch")
        if self.raw_response is None:
            if self.terminal_state in {"success", "partial_success", "malformed_response"}:
                raise ValueError("response-bearing terminal requires raw bytes")
            if self.parsed_record_count or self.parsed_output_sha256 is not None:
                raise ValueError("bodyless terminal cannot claim parser output")
        elif self.parsed_output_sha256 is None:
            raise ValueError("response-bearing terminal requires output digest")
        expected_identity = stable_hash(
            {
                "run_identity": self.run_identity,
                "query_identity": self.query_identity,
                "source": self.source,
                "attempt_identity": self.attempt_identity,
                "request_sequence": self.request_sequence,
            }
        )
        if self.envelope_identity != expected_identity:
            raise ValueError("envelope identity mismatch")
        return self


class ProviderIngestBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract: Literal["provider_ingest_provenance_v1"] = PROVENANCE_CONTRACT
    schema_version: Literal["1"] = SCHEMA_VERSION
    run_identity: OpaqueIdentity
    manifest_identity: OpaqueIdentity
    checkpoint_generation: int = Field(ge=0)
    resource_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    envelopes: list[ProviderAttemptEnvelope]
    authority: Literal["committed_generation_only"] = "committed_generation_only"
    score_scope: Literal[
        "ingest_provenance_only_not_quality_or_official_score"
    ] = "ingest_provenance_only_not_quality_or_official_score"
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_bundle(self) -> "ProviderIngestBundle":
        identities = [item.envelope_identity for item in self.envelopes]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise ValueError("envelopes must be sorted and unique")
        for item in self.envelopes:
            if (
                item.run_identity != self.run_identity
                or item.manifest_identity != self.manifest_identity
                or item.checkpoint_generation != self.checkpoint_generation
            ):
                raise ValueError("envelope run/generation binding mismatch")
        payload = self.model_dump(mode="json")
        payload.pop("bundle_sha256", None)
        if stable_hash(payload) != self.bundle_sha256:
            raise ValueError("bundle digest mismatch")
        return self


class ParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed_record_count: int = Field(ge=0)
    accepted: list[dict[str, Any]]
    rejections: list[RejectedRecord]
    terminal_state: Literal["success", "partial_success", "malformed_response"]
    terminal_reason_code: str | None = None


class ProviderCaptureRecorder:
    """Production-facing observation seam for transport attempts.

    The connector transport supplies the exact bytes it already received; this
    recorder neither performs HTTP nor changes parsing.  Finalization writes the
    two artifacts that must be committed beside the authoritative ledger.
    """

    def __init__(
        self,
        *,
        run_identity: str,
        query_identity: str,
        attempt_identity: str,
        checkpoint_generation: int,
        manifest_identity: str,
        capture_limit_bytes: int | None = None,
    ) -> None:
        self.run_identity = run_identity
        self.query_identity = query_identity
        self.attempt_identity = attempt_identity
        self.checkpoint_generation = checkpoint_generation
        self.manifest_identity = manifest_identity
        if capture_limit_bytes is not None and capture_limit_bytes <= 0:
            raise ProviderIngestError("capture_limit_invalid")
        self.capture_limit_bytes = capture_limit_bytes
        self._captured: list[tuple[ProviderAttemptEnvelope, bytes | None]] = []
        self._request_sequences: set[int] = set()

    def record_attempt(
        self,
        *,
        source: SourceName,
        request_sequence: int,
        resource_operation_identity: str,
        parser_name: ParserName,
        raw_bytes: bytes | None,
        http_status: int | None,
        content_type: str | None,
        encoding: EncodingMetadata,
        compression: Literal["identity", "gzip", "not_available"],
        terminal_state: TerminalState,
        terminal_reason_code: str | None = None,
        page_index: int = 0,
        request_cursor_sha256: str | None = None,
        previous_envelope_identity: str | None = None,
        next_cursor_sha256: str | None = None,
    ) -> ProviderAttemptEnvelope:
        if request_sequence in self._request_sequences:
            raise ProviderIngestError("duplicate_request_sequence")
        if (
            raw_bytes is not None
            and self.capture_limit_bytes is not None
            and len(raw_bytes) > self.capture_limit_bytes
        ):
            # Exact replay evidence is all-or-nothing.  Never truncate an
            # oversized provider response and never present the attempt as a
            # successful parser replay.
            raw_bytes = None
            terminal_state = "capture_size_exceeded"
            terminal_reason_code = "capture_size_exceeded"
        envelope, body = create_envelope(
            run_identity=self.run_identity,
            query_identity=self.query_identity,
            source=source,
            attempt_identity=self.attempt_identity,
            request_sequence=request_sequence,
            resource_operation_identity=resource_operation_identity,
            checkpoint_generation=self.checkpoint_generation,
            manifest_identity=self.manifest_identity,
            parser_name=parser_name,
            raw_bytes=raw_bytes,
            http_status=http_status,
            content_type=content_type,
            encoding=encoding,
            compression=compression,
            terminal_state=terminal_state,
            terminal_reason_code=terminal_reason_code,
            page_index=page_index,
            request_cursor_sha256=request_cursor_sha256,
            previous_envelope_identity=previous_envelope_identity,
            next_cursor_sha256=next_cursor_sha256,
        )
        self._request_sequences.add(request_sequence)
        self._captured.append((envelope, body))
        return envelope

    def finalize(
        self,
        *,
        bundle_path: Path,
        archive_path: Path,
        resource_ledger_path: Path,
    ) -> ProviderIngestBundle:
        return write_capture_bundle(
            bundle_path,
            archive_path,
            run_identity=self.run_identity,
            manifest_identity=self.manifest_identity,
            checkpoint_generation=self.checkpoint_generation,
            resource_ledger_path=resource_ledger_path,
            captured=self._captured,
        )


def opaque_identity(kind: str, *values: object) -> str:
    return stable_hash({"kind": kind, "values": [str(value) for value in values]})


def parse_provider_bytes(
    source: SourceName,
    parser_name: ParserName,
    raw_bytes: bytes,
    *,
    encoding: EncodingMetadata,
    compression: Literal["identity", "gzip", "not_available"],
) -> ParseResult:
    """Replay exact bytes through the same record parsers used by connectors."""

    try:
        decoded_bytes = gzip.decompress(raw_bytes) if compression == "gzip" else raw_bytes
    except (OSError, EOFError):
        return _malformed("compression_decode_failed")
    if compression == "not_available":
        return _malformed("compression_not_available")
    codec = encoding.value if encoding.state == "known" else None
    if not codec:
        return _malformed("encoding_not_available")
    try:
        text = decoded_bytes.decode(codec, errors="strict")
    except (LookupError, UnicodeDecodeError):
        return _malformed("encoding_decode_failed")

    if parser_name == "openalex_search":
        return _parse_json_papers(text, source, "results", _parse_work)
    if parser_name == "semantic_scholar_search":
        return _parse_json_papers(text, source, "data", _parse_paper)
    if parser_name == "pubmed_esearch":
        return _parse_pubmed_ids(text, source)
    if parser_name == "arxiv_atom_search":
        return _parse_xml_papers(text, source, f"{ATOM_NS}entry", _parse_entry)
    if parser_name == "pubmed_efetch":
        return _parse_xml_papers(text, source, ".//PubmedArticle", _parse_article)
    raise ProviderIngestError("parser_not_registered")


def create_envelope(
    *,
    run_identity: str,
    query_identity: str,
    source: SourceName,
    attempt_identity: str,
    request_sequence: int,
    resource_operation_identity: str,
    checkpoint_generation: int,
    manifest_identity: str,
    parser_name: ParserName,
    raw_bytes: bytes | None,
    http_status: int | None,
    content_type: str | None,
    encoding: EncodingMetadata,
    compression: Literal["identity", "gzip", "not_available"],
    terminal_state: TerminalState,
    terminal_reason_code: str | None = None,
    page_index: int = 0,
    request_cursor_sha256: str | None = None,
    previous_envelope_identity: str | None = None,
    next_cursor_sha256: str | None = None,
) -> tuple[ProviderAttemptEnvelope, bytes | None]:
    """Create one deterministic envelope without persisting request secrets."""

    envelope_identity = stable_hash(
        {
            "run_identity": run_identity,
            "query_identity": query_identity,
            "source": source,
            "attempt_identity": attempt_identity,
            "request_sequence": request_sequence,
        }
    )
    parsed_count = accepted_count = 0
    rejections: list[RejectedRecord] = []
    output_digest = None
    raw_ref = None
    resolved_terminal = terminal_state
    if raw_bytes is not None:
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        raw_ref = RawResponseRef(
            member=f"raw/{raw_digest}.bin",
            size_bytes=len(raw_bytes),
            sha256=raw_digest,
        )
        replay = parse_provider_bytes(
            source,
            parser_name,
            raw_bytes,
            encoding=encoding,
            compression=compression,
        )
        parsed_count = replay.parsed_record_count
        accepted_count = len(replay.accepted)
        rejections = replay.rejections
        output_digest = stable_hash(replay.accepted)
        if terminal_state in {"success", "partial_success"}:
            resolved_terminal = replay.terminal_state
            terminal_reason_code = replay.terminal_reason_code
    envelope = ProviderAttemptEnvelope(
        envelope_identity=envelope_identity,
        run_identity=run_identity,
        query_identity=query_identity,
        source=source,
        attempt_identity=attempt_identity,
        request_sequence=request_sequence,
        resource_operation_identity=resource_operation_identity,
        checkpoint_generation=checkpoint_generation,
        manifest_identity=manifest_identity,
        http_status=http_status,
        content_type=_media_type(content_type),
        encoding=encoding,
        compression=compression,
        raw_response=raw_ref,
        page_index=page_index,
        request_cursor_sha256=request_cursor_sha256,
        previous_envelope_identity=previous_envelope_identity,
        next_cursor_sha256=next_cursor_sha256,
        parser_name=parser_name,
        parsed_record_count=parsed_count,
        accepted_record_count=accepted_count,
        rejected_record_count=len(rejections),
        rejections=rejections,
        parsed_output_sha256=output_digest,
        terminal_state=resolved_terminal,
        terminal_reason_code=terminal_reason_code,
    )
    return envelope, raw_bytes


def write_capture_bundle(
    bundle_path: Path,
    archive_path: Path,
    *,
    run_identity: str,
    manifest_identity: str,
    checkpoint_generation: int,
    resource_ledger_path: Path,
    captured: Sequence[tuple[ProviderAttemptEnvelope, bytes | None]],
) -> ProviderIngestBundle:
    """Atomically write the JSON envelope set and deterministic raw archive."""

    raw_by_member: dict[str, bytes] = {}
    envelopes: list[ProviderAttemptEnvelope] = []
    for envelope, raw in captured:
        envelopes.append(envelope)
        if envelope.raw_response is not None:
            if raw is None or hashlib.sha256(raw).hexdigest() != envelope.raw_response.sha256:
                raise ProviderIngestError("raw_response_capture_mismatch")
            existing = raw_by_member.setdefault(envelope.raw_response.member, raw)
            if existing != raw:
                raise ProviderIngestError("content_address_collision")
    archive = _deterministic_raw_archive(raw_by_member)
    durable_atomic_write_bytes(archive_path, archive)
    payload: dict[str, Any] = {
        "contract": PROVENANCE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "run_identity": run_identity,
        "manifest_identity": manifest_identity,
        "checkpoint_generation": checkpoint_generation,
        "resource_ledger_sha256": sha256_file(resource_ledger_path),
        "raw_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "envelopes": [
            item.model_dump(mode="json")
            for item in sorted(envelopes, key=lambda value: value.envelope_identity)
        ],
        "authority": "committed_generation_only",
        "score_scope": "ingest_provenance_only_not_quality_or_official_score",
    }
    payload["bundle_sha256"] = stable_hash(payload)
    bundle = ProviderIngestBundle.model_validate(payload)
    durable_atomic_write_bytes(bundle_path, stable_json_bytes(bundle.model_dump(mode="json")))
    return bundle


def verify_capture_bundle(
    bundle_path: Path,
    archive_path: Path,
    *,
    resource_ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Verify bytes, parser replay, pagination and optional ledger binding."""

    violations: list[dict[str, Any]] = []
    try:
        bundle = ProviderIngestBundle.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
        archive_bytes = archive_path.read_bytes()
    except (OSError, UnicodeError, ValidationError, ValueError) as exc:
        return _report([_violation("bundle_or_archive_unreadable", "$", type(exc).__name__)])
    if hashlib.sha256(archive_bytes).hexdigest() != bundle.raw_archive_sha256:
        violations.append(_violation("raw_archive_hash_mismatch", "$.raw_archive_sha256"))
    try:
        raw_members = _read_raw_archive(archive_bytes)
    except ProviderIngestError as exc:
        violations.append(_violation(str(exc), "$.raw_archive"))
        raw_members = {}

    operations: dict[str, Any] = {}
    if resource_ledger_path is not None:
        try:
            if sha256_file(resource_ledger_path) != bundle.resource_ledger_sha256:
                violations.append(_violation("resource_ledger_hash_mismatch", "$.resource_ledger_sha256"))
            ledger = ResourceLedgerV1.model_validate_json(
                resource_ledger_path.read_text(encoding="utf-8")
            )
            ledger_report = validate_resource_ledger(ledger)
            if ledger_report["status"] != "passed":
                violations.append(
                    _violation("resource_ledger_accounting_invalid", "$.resource_ledger")
                )
            if (
                ledger.run_identity != bundle.run_identity
                or ledger.manifest_identity != bundle.manifest_identity
            ):
                violations.append(_violation("resource_ledger_run_binding_mismatch", "$.resource_ledger"))
            operations = {
                operation.operation_identity: operation
                for query in ledger.queries
                for operation in query.operations
                if operation.operation_type == "adapter_call"
            }
        except (OSError, UnicodeError, ValidationError, ValueError) as exc:
            violations.append(_violation("resource_ledger_unreadable", "$.resource_ledger", type(exc).__name__))

    consumed_members: Counter[str] = Counter()
    envelopes_by_chain: dict[
        tuple[str, str, str, str], list[ProviderAttemptEnvelope]
    ] = defaultdict(list)
    operation_counts: Counter[str] = Counter()
    for index, envelope in enumerate(bundle.envelopes):
        location = f"$.envelopes[{index}]"
        operation_counts[envelope.resource_operation_identity] += 1
        envelopes_by_chain[
            (
                envelope.query_identity,
                envelope.source,
                envelope.attempt_identity,
                envelope.parser_name,
            )
        ].append(envelope)
        if operations:
            operation = operations.get(envelope.resource_operation_identity)
            if operation is None:
                violations.append(_violation("resource_operation_missing", location + ".resource_operation_identity"))
            else:
                from scholar_agent.evaluation.resource_accounting import (  # noqa: PLC0415
                    opaque_resource_identity,
                )

                if (
                    operation.query_identity != envelope.query_identity
                    or operation.attempt_identity != envelope.attempt_identity
                    or operation.checkpoint_generation
                    != envelope.checkpoint_generation
                    or operation.manifest_identity != envelope.manifest_identity
                    or operation.source_identity
                    != opaque_resource_identity("source", envelope.source)
                ):
                    violations.append(
                        _violation("resource_operation_binding_mismatch", location)
                    )
        if envelope.raw_response is None:
            continue
        consumed_members[envelope.raw_response.member] += 1
        raw = raw_members.get(envelope.raw_response.member)
        if raw is None:
            violations.append(_violation("raw_response_member_missing", location + ".raw_response"))
            continue
        if len(raw) != envelope.raw_response.size_bytes or hashlib.sha256(raw).hexdigest() != envelope.raw_response.sha256:
            violations.append(_violation("raw_response_hash_or_size_mismatch", location + ".raw_response"))
            continue
        replay = parse_provider_bytes(
            envelope.source,
            envelope.parser_name,
            raw,
            encoding=envelope.encoding,
            compression=envelope.compression,
        )
        observed = {
            "parsed": replay.parsed_record_count,
            "accepted": len(replay.accepted),
            "rejected": len(replay.rejections),
            "output_sha256": stable_hash(replay.accepted),
            "terminal": replay.terminal_state,
            "reason": replay.terminal_reason_code,
        }
        expected = {
            "parsed": envelope.parsed_record_count,
            "accepted": envelope.accepted_record_count,
            "rejected": envelope.rejected_record_count,
            "output_sha256": envelope.parsed_output_sha256,
            "terminal": envelope.terminal_state,
            "reason": envelope.terminal_reason_code,
        }
        if observed != expected:
            violations.append(_violation("parser_replay_mismatch", location, observed, expected))

    if set(raw_members) != set(consumed_members):
        violations.append(_violation("raw_archive_inventory_mismatch", "$.raw_archive.inventory"))
    _validate_pagination(envelopes_by_chain, violations)
    if operations:
        for identity, operation in sorted(operations.items()):
            expected = (
                int(operation.api_request_count.value)
                if operation.api_request_count.state == "known"
                else None
            )
            if expected is None:
                violations.append(_violation("adapter_request_count_not_available", "$.resource_ledger.operations"))
            elif operation_counts[identity] != expected:
                violations.append(
                    _violation(
                        "adapter_envelope_count_mismatch",
                        "$.resource_ledger.operations",
                        operation_counts[identity],
                        expected,
                    )
                )
        unknown = sorted(set(operation_counts) - set(operations))
        if unknown:
            violations.append(_violation("unregistered_envelope_operation", "$.envelopes"))
    return _report(violations, envelope_count=len(bundle.envelopes), raw_member_count=len(raw_members))


def replay_capture_bundle(bundle_path: Path, archive_path: Path) -> dict[str, Any]:
    report = verify_capture_bundle(bundle_path, archive_path)
    return {
        **report,
        "stage": "replay_parser",
        "replay_scope": "production_parser_only_not_retrieval_quality",
    }


def audit_frozen_record162(repository_root: Path) -> dict[str, Any]:
    """Fail closed: legacy Record160/162 never retained parser-pre bytes."""

    audit = repository_root / "benchmark" / "run_provenance_legacy_audit.json"
    if not audit.is_file():
        raise ProviderIngestNotEligible("legacy_audit_missing")
    return {
        "protocol": PROVENANCE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "reason_code": "missing_parser_pre_raw_response_bytes",
        "historical_artifacts_modified": False,
        "raw_payload_inferred": False,
        "execution": _execution_zero(),
    }


def deterministic_fixture_matrix(root: Path) -> dict[str, Any]:
    """Exercise all offline source and failure classes with synthetic bytes."""

    root.mkdir(parents=True, exist_ok=True)
    run = opaque_identity("run", "fixture")
    query = opaque_identity("query", "fixture")
    attempt = opaque_identity("attempt", "fixture")
    manifest = opaque_identity("manifest", "fixture")
    encoding = EncodingMetadata(state="known", value="utf-8")
    fixtures: list[
        tuple[
            str,
            SourceName,
            ParserName,
            bytes | None,
            int | None,
            TerminalState,
            str,
            Literal["identity", "gzip", "not_available"],
        ]
    ] = [
        ("openalex_success", "openalex", "openalex_search", _openalex_fixture(), 200, "success", "application/json", "identity"),
        ("semantic_success", "semantic_scholar", "semantic_scholar_search", _semantic_fixture(), 200, "success", "application/json", "identity"),
        ("semantic_gzip", "semantic_scholar", "semantic_scholar_search", gzip.compress(_semantic_fixture(), mtime=0), 200, "success", "application/json", "gzip"),
        ("arxiv_success", "arxiv", "arxiv_atom_search", _arxiv_fixture(), 200, "success", "application/atom+xml", "identity"),
        ("pubmed_search_success", "pubmed", "pubmed_esearch", _pubmed_search_fixture(), 200, "success", "application/json", "identity"),
        ("pubmed_fetch_success", "pubmed", "pubmed_efetch", _pubmed_fetch_fixture(), 200, "success", "application/xml", "identity"),
        ("duplicate_records", "openalex", "openalex_search", _openalex_duplicate_fixture(), 200, "success", "application/json", "identity"),
        ("partial_page", "openalex", "openalex_search", _openalex_partial_fixture(), 200, "partial_success", "application/json", "identity"),
        ("malformed_json", "openalex", "openalex_search", b'{"results":[', 200, "success", "application/json", "identity"),
        ("unknown_schema", "semantic_scholar", "semantic_scholar_search", b'{"unexpected":[]}', 200, "success", "application/json", "identity"),
        ("illegal_encoding", "openalex", "openalex_search", b"\xff\xfe", 200, "success", "application/json", "identity"),
        ("truncated_xml", "arxiv", "arxiv_atom_search", b"<feed><entry>", 200, "success", "application/atom+xml", "identity"),
        ("empty_response", "openalex", "openalex_search", b'{"results":[]}', 200, "success", "application/json", "identity"),
        ("rate_limited", "semantic_scholar", "semantic_scholar_search", None, 429, "rate_limited", "application/json", "not_available"),
        ("provider_503", "openalex", "openalex_search", None, 503, "http_error", "application/json", "not_available"),
        ("timeout", "arxiv", "arxiv_atom_search", None, None, "timeout", "application/atom+xml", "not_available"),
        ("adapter_exception", "pubmed", "pubmed_efetch", None, None, "adapter_exception", "application/xml", "not_available"),
    ]
    captured: list[tuple[ProviderAttemptEnvelope, bytes | None]] = []
    scenario_rows: list[dict[str, Any]] = []
    chain_totals = Counter((source, parser) for _, source, parser, *_ in fixtures)
    chain_indexes: Counter[tuple[str, str]] = Counter()
    previous_by_chain: dict[tuple[str, str], str] = {}
    for sequence, (
        name,
        source,
        parser,
        raw,
        status,
        terminal,
        content_type,
        compression,
    ) in enumerate(fixtures):
        chain = (source, parser)
        page_index = chain_indexes[chain]
        request_cursor = stable_hash({"source": source, "parser": parser, "page": page_index})
        next_cursor = (
            stable_hash({"source": source, "parser": parser, "page": page_index + 1})
            if page_index + 1 < chain_totals[chain]
            else None
        )
        item, body = create_envelope(
            run_identity=run,
            query_identity=query,
            source=source,
            attempt_identity=attempt,
            request_sequence=sequence,
            resource_operation_identity=opaque_identity("operation", sequence),
            checkpoint_generation=1,
            manifest_identity=manifest,
            parser_name=parser,
            raw_bytes=raw,
            http_status=status,
            content_type=content_type,
            encoding=encoding,
            compression=compression,
            terminal_state=terminal,
            terminal_reason_code=name if raw is None else None,
            page_index=page_index,
            request_cursor_sha256=request_cursor,
            previous_envelope_identity=previous_by_chain.get(chain),
            next_cursor_sha256=next_cursor,
        )
        chain_indexes[chain] += 1
        previous_by_chain[chain] = item.envelope_identity
        captured.append((item, body))
        scenario_rows.append(
            {
                "scenario": name,
                "source": source,
                "terminal_state": item.terminal_state,
                "accepted_record_count": item.accepted_record_count,
                "rejected_record_count": item.rejected_record_count,
            }
        )
    ledger_path = root / "resource_ledger.json"
    durable_atomic_write_bytes(
        ledger_path,
        stable_json_bytes(_fixture_ledger(run, query, attempt, manifest, captured)),
    )
    bundle_path = root / BUNDLE_NAME
    archive_path = root / RAW_ARCHIVE_NAME
    write_capture_bundle(
        bundle_path,
        archive_path,
        run_identity=run,
        manifest_identity=manifest,
        checkpoint_generation=1,
        resource_ledger_path=ledger_path,
        captured=captured,
    )
    verification = verify_capture_bundle(bundle_path, archive_path, resource_ledger_path=ledger_path)
    return {
        "protocol": PROVENANCE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if verification["exit_code"] == 0 else "provenance_or_parser_violation",
        "exit_code": verification["exit_code"],
        "scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
        "verification": verification,
        "execution": _execution_zero(),
    }


def _parse_json_papers(text: str, source: str, field: str, parser: Any) -> ParseResult:
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, UnicodeError):
        return _malformed("malformed_document")
    if not isinstance(document, dict) or not isinstance(document.get(field), list):
        return _unknown_schema()
    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRecord] = []
    values = document[field]
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            rejected.append(_rejection(index, "record_not_object", value))
            continue
        try:
            paper = parser(value)
        except Exception:  # noqa: BLE001 - mirrors connector record isolation
            rejected.append(_rejection(index, "malformed_record", value))
            continue
        if paper is None:
            rejected.append(_rejection(index, "missing_required_field", value))
            continue
        accepted.append(paper.model_dump(mode="json"))
    return _parsed(values, accepted, rejected)


def _parse_pubmed_ids(text: str, source: str) -> ParseResult:
    try:
        document = json.loads(text)
        values = document["esearchresult"]["idlist"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return _unknown_schema()
    if not isinstance(values, list):
        return _unknown_schema()
    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRecord] = []
    for index, value in enumerate(values):
        normalized = _normalize_pmid(value)
        if normalized:
            accepted.append({"pubmed_id": normalized, "sources": [source]})
        else:
            rejected.append(_rejection(index, "identity_invalid", value))
    return _parsed(values, accepted, rejected)


def _parse_xml_papers(text: str, source: str, selector: str, parser: Any) -> ParseResult:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _malformed("malformed_document")
    values = root.findall(selector)
    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRecord] = []
    for index, value in enumerate(values):
        try:
            paper = parser(value)
        except Exception:  # noqa: BLE001 - mirrors connector record isolation
            rejected.append(_rejection(index, "malformed_record", ET.tostring(value)))
            continue
        if paper is None:
            rejected.append(_rejection(index, "missing_required_field", ET.tostring(value)))
            continue
        accepted.append(paper.model_dump(mode="json"))
    return _parsed(values, accepted, rejected)


def _parsed(values: Sequence[Any], accepted: list[dict[str, Any]], rejected: list[RejectedRecord]) -> ParseResult:
    terminal = "partial_success" if rejected and accepted else "success"
    if rejected and not accepted:
        terminal = "partial_success"
    return ParseResult(
        parsed_record_count=len(values),
        accepted=accepted,
        rejections=rejected,
        terminal_state=terminal,
        terminal_reason_code="record_rejections_present" if rejected else None,
    )


def _malformed(reason: str) -> ParseResult:
    return ParseResult(
        parsed_record_count=0,
        accepted=[],
        rejections=[],
        terminal_state="malformed_response",
        terminal_reason_code=reason,
    )


def _unknown_schema() -> ParseResult:
    return ParseResult(
        parsed_record_count=1,
        accepted=[],
        rejections=[RejectedRecord(record_index=None, reason_code="unknown_schema")],
        terminal_state="partial_success",
        terminal_reason_code="unknown_schema",
    )


def _rejection(index: int, reason: ReasonCode, value: Any) -> RejectedRecord:
    if isinstance(value, bytes):
        evidence = hashlib.sha256(value).hexdigest()
    else:
        try:
            evidence = stable_hash(value)
        except TypeError:
            evidence = stable_hash(str(type(value).__name__))
    return RejectedRecord(record_index=index, reason_code=reason, evidence_sha256=evidence)


def _deterministic_raw_archive(raw_by_member: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member in sorted(raw_by_member):
            _validate_raw_member(member)
            content = bytes(raw_by_member[member])
            info = tarfile.TarInfo(member)
            info.size = len(content)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o400
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _read_raw_archive(content: bytes) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    total = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) > MAX_RAW_MEMBERS:
                raise ProviderIngestError("raw_archive_member_limit_exceeded")
            for item in members:
                _validate_raw_member(item.name)
                if not item.isfile() or item.issym() or item.islnk():
                    raise ProviderIngestError("raw_archive_non_regular_member")
                if item.name in values:
                    raise ProviderIngestError("raw_archive_duplicate_member")
                if item.size > MAX_RAW_MEMBER_BYTES:
                    raise ProviderIngestError("raw_archive_member_size_exceeded")
                total += item.size
                if total > MAX_RAW_TOTAL_BYTES:
                    raise ProviderIngestError("raw_archive_total_size_exceeded")
                stream = archive.extractfile(item)
                if stream is None:
                    raise ProviderIngestError("raw_archive_member_unreadable")
                values[item.name] = stream.read()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ProviderIngestError("raw_archive_malformed") from exc
    return values


def _validate_raw_member(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "raw" or ".." in path.parts:
        raise ProviderIngestError("raw_archive_path_invalid")
    expected = path.parts[1]
    if re.fullmatch(r"[0-9a-f]{64}\.bin", expected) is None:
        raise ProviderIngestError("raw_archive_member_name_invalid")


def _validate_pagination(
    chains: Mapping[tuple[str, str, str, str], list[ProviderAttemptEnvelope]],
    violations: list[dict[str, Any]],
) -> None:
    for key, values in sorted(chains.items()):
        pages = sorted(values, key=lambda item: item.page_index)
        if [item.page_index for item in pages] != list(range(len(pages))):
            violations.append(_violation("pagination_index_gap", "$.envelopes.pagination"))
        seen_cursors: set[str] = set()
        for index, item in enumerate(pages):
            expected_previous = pages[index - 1].envelope_identity if index else None
            if item.previous_envelope_identity != expected_previous:
                violations.append(_violation("pagination_chain_broken", "$.envelopes.pagination"))
            if item.request_cursor_sha256:
                if item.request_cursor_sha256 in seen_cursors:
                    violations.append(_violation("pagination_cursor_cycle", "$.envelopes.pagination"))
                seen_cursors.add(item.request_cursor_sha256)
            if index + 1 < len(pages) and item.next_cursor_sha256 != pages[index + 1].request_cursor_sha256:
                violations.append(_violation("pagination_cursor_link_mismatch", "$.envelopes.pagination"))


def _media_type(value: str | None) -> str | None:
    return value.split(";", 1)[0].strip().casefold() if value else None


def _report(violations: list[dict[str, Any]], **counts: Any) -> dict[str, Any]:
    return {
        "protocol": PROVENANCE_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "status": "passed" if not violations else "provenance_or_parser_violation",
        "exit_code": EXIT_PASSED if not violations else EXIT_VIOLATION,
        **counts,
        "violation_count": len(violations),
        "violations": violations,
        "execution": _execution_zero(),
    }


def _violation(invariant: str, path: str, observed: Any = None, expected: Any = None) -> dict[str, Any]:
    return {
        "invariant": invariant,
        "path": path,
        "observed_sha256": stable_hash(observed) if observed is not None else None,
        "expected_sha256": stable_hash(expected) if expected is not None else None,
    }


def _execution_zero() -> dict[str, Any]:
    return {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "gold_or_qrels_loaded": False,
        "quality_metric_count": 0,
    }


def _fixture_ledger(
    run: str,
    query: str,
    attempt: str,
    manifest: str,
    captured: Sequence[tuple[ProviderAttemptEnvelope, bytes | None]],
) -> dict[str, Any]:
    from scholar_agent.evaluation.resource_accounting import (  # noqa: PLC0415
        BudgetVector,
        QueryResourceLedger,
        ResourceOperation,
        _aggregate_budget,
        _budget_summary,
        _operation_totals,
        known,
        opaque_resource_identity,
        unavailable,
    )

    request_count = len(captured)

    operations = [
        ResourceOperation(
            operation_identity=opaque_identity("operation", "reservation"),
            run_identity=run,
            query_identity=query,
            attempt_identity=attempt,
            operation_type="budget_reservation",
            request_sequence=0,
            checkpoint_generation=1,
            manifest_identity=manifest,
            budget_reserved=BudgetVector(search_rounds=request_count),
            terminal_status="success",
        )
    ]
    operations.extend(
        [
        ResourceOperation(
            operation_identity=envelope.resource_operation_identity,
            run_identity=run,
            query_identity=query,
            source_identity=opaque_resource_identity("source", envelope.source),
            attempt_identity=attempt,
            operation_type="adapter_call",
            request_sequence=envelope.request_sequence + 1,
            checkpoint_generation=1,
            manifest_identity=manifest,
            budget_consumed=BudgetVector(search_rounds=1),
            api_request_count=known(1, "requests"),
            pagination_count=known(1, "pages"),
            returned_record_count=envelope.accepted_record_count,
            terminal_status=(
                "success"
                if envelope.terminal_state == "success"
                else "partial"
                if envelope.terminal_state == "partial_success"
                else "timeout"
                if envelope.terminal_state == "timeout"
                else "rate_limited"
                if envelope.terminal_state == "rate_limited"
                else "failed"
            ),
            adapter_started=True,
        )
        for envelope, _raw in captured
        ]
    )
    totals = _operation_totals(operations)
    budget = _budget_summary(
        operations,
        limits=BudgetVector(search_rounds=request_count),
        latency_limit_seconds=1,
        elapsed_seconds=unavailable("seconds"),
    )
    query_ledger = QueryResourceLedger(
        run_identity=run,
        query_identity=query,
        attempt_identity=attempt,
        checkpoint_generation=1,
        manifest_identity=manifest,
        query_terminal_status="succeeded",
        operations=operations,
        totals=totals,
        budget=budget,
    )
    ledger = ResourceLedgerV1(
        run_identity=run,
        manifest_identity=manifest,
        expected_query_identities=[query],
        queries=[query_ledger],
        totals=totals,
        budget=_aggregate_budget([budget]),
        selected_attempts={query: attempt},
    )
    return ledger.model_dump(mode="json")


def _openalex_fixture() -> bytes:
    return stable_json_bytes({"results": [{"id": "https://openalex.org/W1", "title": "Synthetic OpenAlex", "publication_year": 2024, "authorships": []}]}, indent=None)


def _semantic_fixture() -> bytes:
    return stable_json_bytes({"data": [{"paperId": "S2-1", "title": "Synthetic Semantic", "authors": [], "year": 2024}]}, indent=None)


def _openalex_duplicate_fixture() -> bytes:
    item = {"id": "https://openalex.org/W2", "title": "Synthetic Duplicate", "publication_year": 2024, "authorships": []}
    return stable_json_bytes({"results": [item, item]}, indent=None)


def _openalex_partial_fixture() -> bytes:
    item = {"id": "https://openalex.org/W3", "title": "Synthetic Partial", "publication_year": 2024, "authorships": []}
    return stable_json_bytes({"results": [item, "not-an-object"]}, indent=None)


def _arxiv_fixture() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2401.00001</id><title>Synthetic arXiv</title><summary>Offline fixture.</summary><published>2024-01-01T00:00:00Z</published><author><name>Fixture Author</name></author></entry></feed>'''


def _pubmed_search_fixture() -> bytes:
    return b'{"esearchresult":{"idlist":["12345"]}}'


def _pubmed_fetch_fixture() -> bytes:
    return b'''<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>Synthetic PubMed</ArticleTitle><Abstract><AbstractText>Offline fixture.</AbstractText></Abstract><Journal><Title>Fixture Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal><AuthorList><Author><ForeName>Fixture</ForeName><LastName>Author</LastName></Author></AuthorList></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>'''
