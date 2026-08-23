from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation.release_authenticity_signing import (
    NAMESPACE,
    REAL_BLOCKERS,
    SOURCE_COMMIT,
    AuthenticityError,
    AuthenticityNotReady,
    add_rotated_key,
    audit_current,
    build_envelope,
    canonical_json,
    empty_trust_root,
    finalize_trust_root,
    generate_test_key,
    load_protocol,
    register_key,
    sha256_file,
    sign_envelope,
    simulate_matrix,
    stable_hash,
    transition_key,
    verify_issuance_set,
    verify_signature_package,
    verify_trust_root,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmark/release_authenticity_signing_v1_protocol.json"
TRUST_ROOT = (
    ROOT / "benchmark/release_authenticity_signing_v1_trust_root.json"
)
SCRIPT = ROOT / "scripts/check_release_authenticity.py"
TRANSPARENCY_ROOT = (
    "2d2fee76aa56f48a7e4a204bdbb16995701f7f930abfa216e962af04bfb06473"
)


def _key(tmp_path: Path, identity: str = "test:alpha") -> tuple[Path, dict]:
    private_key, public_key, _fingerprint = generate_test_key(
        tmp_path / identity.replace(":", "-"),
        key_identity=identity,
    )
    trust_root = register_key(
        empty_trust_root(),
        key_identity=identity,
        public_key=public_key,
        test_only=True,
    )
    return private_key, trust_root


def _artifact(tmp_path: Path, content: bytes = b"release-candidate\n") -> Path:
    path = tmp_path / "artifact.bin"
    path.write_bytes(content)
    return path


def _envelope(
    artifact: Path,
    *,
    identity: str = "test:alpha",
    commit: str = SOURCE_COMMIT,
    artifact_type: str = "release_candidate",
    test_only: bool = True,
) -> dict:
    return build_envelope(
        artifact_type=artifact_type,
        artifact_version="v1",
        content_sha256=sha256_file(artifact),
        transparency_root=TRANSPARENCY_ROOT,
        transparency_sequence=0,
        code_commit=commit,
        readiness_status="ready_with_declared_blockers",
        key_identity=identity,
        test_only=test_only,
    )


def _package(tmp_path: Path) -> tuple[Path, dict, dict]:
    artifact = _artifact(tmp_path)
    private_key, trust_root = _key(tmp_path)
    package = sign_envelope(
        _envelope(artifact),
        private_key=private_key,
        trust_root=trust_root,
        issuance_sequence=0,
    )
    return artifact, trust_root, package


def _rehash_package(package: dict) -> dict:
    value = copy.deepcopy(package)
    value["package_sha256"] = "0" * 64
    value["package_sha256"] = stable_hash(value)
    return value


def _run(
    *arguments: str,
    cwd: Path = ROOT,
    home: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        "HOME": str(home or cwd),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    for key in (
        "SystemRoot",
        "WINDIR",
        "PROGRAMDATA",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "TEMP",
        "TMP",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_protocol_and_current_empty_trust_root_are_closed() -> None:
    protocol = load_protocol(PROTOCOL)
    trust = verify_trust_root(json.loads(TRUST_ROOT.read_text()))
    report = audit_current(ROOT, protocol, json.loads(TRUST_ROOT.read_text()))
    assert trust["key_count"] == 0
    assert trust["active_real_key_count"] == 0
    assert report["exit_code"] == 3
    assert report["status"] == "not_ready_missing_real_trust_anchor_or_signer"
    assert report["formal_blockers"] == list(REAL_BLOCKERS)
    assert report["formal_validation_complete"] is False


def test_protocol_security_policy_drift_is_rejected(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text())
    protocol["artifact_types"].remove("clearance_receipt")
    protocol["protocol_sha256"] = "0" * 64
    protocol["protocol_sha256"] = stable_hash(protocol)
    path = tmp_path / "drifted-protocol.json"
    path.write_bytes(canonical_json(protocol))
    with pytest.raises(AuthenticityError, match="protocol_schema_invalid"):
        load_protocol(path)


def test_valid_signing_and_verification_are_deterministic(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    private_key, trust_root = _key(tmp_path)
    envelope = _envelope(artifact)
    first = sign_envelope(
        envelope,
        private_key=private_key,
        trust_root=trust_root,
        issuance_sequence=0,
    )
    second = sign_envelope(
        envelope,
        private_key=private_key,
        trust_root=trust_root,
        issuance_sequence=0,
    )
    assert canonical_json(first) == canonical_json(second)
    report = verify_signature_package(
        first,
        trust_root=trust_root,
        artifact_sha256=sha256_file(artifact),
    )
    assert report["signature_verified"] is True
    assert report["test_only"] is True


def test_two_environment_cli_verification_is_identical(tmp_path: Path) -> None:
    artifact, trust_root, package = _package(tmp_path)
    trust_path = tmp_path / "trust.json"
    package_path = tmp_path / "package.json"
    trust_path.write_bytes(canonical_json(trust_root))
    package_path.write_bytes(canonical_json(package))
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _run(
        "verify",
        "--artifact",
        str(artifact),
        "--signature-package",
        str(package_path),
        "--trust-root",
        str(trust_path),
        cwd=first_dir,
        home=first_dir,
    )
    second = _run(
        "verify",
        "--artifact",
        str(artifact),
        "--signature-package",
        str(package_path),
        "--trust-root",
        str(trust_path),
        cwd=second_dir,
        home=second_dir,
    )
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda package: package["envelope"].__setitem__(
                "code_commit", "1" * 40
            ),
            "signature",
        ),
        (
            lambda package: package["envelope"]["artifact"].__setitem__(
                "type", "standalone_auditor_bundle"
            ),
            "signature",
        ),
        (
            lambda package: package["envelope"]["transparency_log"].__setitem__(
                "root_sha256", "1" * 64
            ),
            "signature",
        ),
        (
            lambda package: package["envelope"]["signing"].__setitem__(
                "namespace", "wrong-namespace"
            ),
            "envelope",
        ),
        (
            lambda package: package["envelope"].__setitem__(
                "formal_blockers", []
            ),
            "envelope",
        ),
    ],
)
def test_signed_context_tampering_is_rejected(
    tmp_path: Path, mutation, reason: str
) -> None:
    artifact, trust_root, package = _package(tmp_path)
    mutation(package)
    package["envelope_sha256"] = stable_hash(package["envelope"])
    package = _rehash_package(package)
    with pytest.raises(AuthenticityError, match=reason):
        verify_signature_package(
            package,
            trust_root=trust_root,
            artifact_sha256=sha256_file(artifact),
        )


def test_artifact_content_tampering_is_rejected(tmp_path: Path) -> None:
    artifact, trust_root, package = _package(tmp_path)
    artifact.write_bytes(b"tampered\n")
    with pytest.raises(AuthenticityError, match="binding"):
        verify_signature_package(
            package,
            trust_root=trust_root,
            artifact_sha256=sha256_file(artifact),
        )


def test_unknown_key_and_public_key_replacement_are_rejected(
    tmp_path: Path,
) -> None:
    artifact, trust_root, package = _package(tmp_path)
    unknown = copy.deepcopy(package)
    unknown["key_identity"] = "test:unknown"
    unknown = _rehash_package(unknown)
    with pytest.raises(AuthenticityError):
        verify_signature_package(
            unknown,
            trust_root=trust_root,
            artifact_sha256=sha256_file(artifact),
        )

    _other_private, other_public, _fingerprint = generate_test_key(
        tmp_path / "other",
        key_identity="test:other",
    )
    replaced = copy.deepcopy(trust_root)
    replaced["keys"][0]["public_key"] = other_public
    replaced = finalize_trust_root(replaced)
    with pytest.raises(AuthenticityError, match="key_schema"):
        verify_trust_root(replaced)


def test_rotation_preserves_historical_verification_and_blocks_old_signing(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    old_private, trust_root = _key(tmp_path, "test:old")
    historical = sign_envelope(
        _envelope(artifact, identity="test:old"),
        private_key=old_private,
        trust_root=trust_root,
        issuance_sequence=0,
    )
    new_private, new_public, _fingerprint = generate_test_key(
        tmp_path / "new-key",
        key_identity="test:new",
    )
    rotated = transition_key(
        trust_root,
        key_identity="test:old",
        to_state="rotated",
        retired_sequence=10,
        superseded_by="test:new",
        signer_private_key=old_private,
    )
    rotated = add_rotated_key(
        rotated,
        key_identity="test:new",
        public_key=new_public,
        test_only=True,
        activated_sequence=10,
    )
    verify_signature_package(
        historical,
        trust_root=rotated,
        artifact_sha256=sha256_file(artifact),
    )
    with pytest.raises(AuthenticityError, match="not_active"):
        sign_envelope(
            _envelope(artifact, identity="test:old"),
            private_key=old_private,
            trust_root=rotated,
            issuance_sequence=10,
        )
    new_package = sign_envelope(
        _envelope(artifact, identity="test:new"),
        private_key=new_private,
        trust_root=rotated,
        issuance_sequence=10,
    )
    assert new_package["key_identity"] == "test:new"


def test_revocation_blocks_new_signatures_but_keeps_old_verifiable(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path)
    private_key, trust_root = _key(tmp_path)
    historical = sign_envelope(
        _envelope(artifact),
        private_key=private_key,
        trust_root=trust_root,
        issuance_sequence=0,
    )
    revoked = transition_key(
        trust_root,
        key_identity="test:alpha",
        to_state="revoked",
        retired_sequence=5,
        signer_private_key=private_key,
    )
    verify_signature_package(
        historical,
        trust_root=revoked,
        artifact_sha256=sha256_file(artifact),
    )
    with pytest.raises(AuthenticityError, match="not_active"):
        sign_envelope(
            _envelope(artifact),
            private_key=private_key,
            trust_root=revoked,
            issuance_sequence=5,
        )


def test_transition_claim_without_valid_old_key_signature_is_rejected(
    tmp_path: Path,
) -> None:
    private_key, trust_root = _key(tmp_path)
    revoked = transition_key(
        trust_root,
        key_identity="test:alpha",
        to_state="revoked",
        retired_sequence=5,
        signer_private_key=private_key,
    )
    tampered = copy.deepcopy(revoked)
    tampered["transitions"][0]["authorization"]["signature_base64"] = "AAAA"
    tampered["transitions"][0]["content_sha256"] = "0" * 64
    tampered["transitions"][0]["content_sha256"] = stable_hash(
        {
            **tampered["transitions"][0],
            "content_sha256": "0" * 64,
        }
    )
    tampered = finalize_trust_root(tampered)
    with pytest.raises(AuthenticityError, match="signature"):
        verify_trust_root(tampered)


def test_test_key_cannot_impersonate_real_release(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path)
    private_key, trust_root = _key(tmp_path)
    with pytest.raises(AuthenticityError, match="test_only"):
        sign_envelope(
            _envelope(artifact, test_only=False),
            private_key=private_key,
            trust_root=trust_root,
            issuance_sequence=0,
        )


def test_duplicate_issuance_conflict_is_rejected(tmp_path: Path) -> None:
    artifact, _trust_root, package = _package(tmp_path)
    conflict = copy.deepcopy(package)
    conflict["envelope"]["artifact"]["content_sha256"] = "1" * 64
    conflict["envelope_sha256"] = stable_hash(conflict["envelope"])
    conflict = _rehash_package(conflict)
    with pytest.raises(AuthenticityError, match="duplicate_issuance"):
        verify_issuance_set([package, conflict])
    verify_issuance_set([package, copy.deepcopy(package)])
    assert artifact.exists()


def test_missing_signing_tool_is_not_ready(tmp_path: Path) -> None:
    with pytest.raises(AuthenticityNotReady, match="ssh_keygen"):
        generate_test_key(
            tmp_path / "missing-tool",
            key_identity="test:missing",
            executable=str(tmp_path / "does-not-exist"),
        )


def test_cli_audit_is_exit_three_without_real_anchor() -> None:
    first = _run("audit-readiness")
    second = _run("audit-readiness")
    assert first.returncode == second.returncode == 3
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    report = json.loads(first.stdout)
    assert report["status"] == "not_ready_missing_real_trust_anchor_or_signer"
    assert report["formal_validation_complete"] is False


def test_cli_generation_does_not_echo_private_key_or_secret_material(
    tmp_path: Path,
) -> None:
    key_dir = tmp_path / "operator-only"
    trust = tmp_path / "trust.json"
    result = _run(
        "generate-test-key",
        "--output-dir",
        str(key_dir),
        "--key-identity",
        "test:cli",
        "--trust-root-output",
        str(trust),
    )
    assert result.returncode == 0
    assert result.stderr == ""
    assert str(key_dir) not in result.stdout
    assert "PRIVATE KEY" not in result.stdout
    if os.name != "nt":
        assert (key_dir / "operator-test-key").stat().st_mode & 0o777 == 0o600
    assert json.loads(result.stdout)["test_only"] is True


def test_private_keys_are_not_tracked() -> None:
    private_key_marker = "BEGIN OPENSSH " + "PRIVATE KEY"
    result = subprocess.run(
        ["git", "grep", "-l", private_key_marker, "--", ":!third_party"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""


def test_synthetic_attack_matrix_is_deterministic() -> None:
    first = simulate_matrix()
    second = simulate_matrix()
    assert canonical_json(first) == canonical_json(second)
    assert first["scenario_count"] == 9
    assert {row["status"] for row in first["scenarios"]} == {"passed"}
