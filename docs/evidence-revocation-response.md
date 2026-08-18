# Evidence revocation and incident response

`evidence_revocation_response_v1` is an offline, append-only control for
invalidating evidence that is later found to be tampered, leaked, defective,
incorrect, stale, over-extrapolated, or wrongly published. It does not modify
or delete historical evidence and does not calculate retrieval quality.

The authoritative real ledger is
`benchmark/evidence_revocation_response_v1_ledger.json`. Its normal state is an
empty event list. Each event binds a structured reason code, opaque operator,
previous and next evidence state, impact scope, trigger digest, and the prior
event hash. `under_investigation`, `revoked`, and `superseded` are release
blocking states. `restored` is accepted only after a `superseded` event names a
fresh, hash-matching replacement evidence item and a current passing gate.
Removing an event never restores validity.

Impact propagation reuses the frozen freshness dependency graph. An active
incident invalidates dependent evidence, claims, read-only gates, the
readiness bundle, standalone auditor bundle, release candidate, and clearance
receipt. Evidence outside the dependency closure remains valid. Incident
bundles contain identities, hashes, minimum rerun gates, and prohibited
publication actions only; they never copy sensitive source material.

Read-only commands:

```bash
PYTHONPATH=src python scripts/check_evidence_revocation.py audit-current
PYTHONPATH=src python scripts/check_evidence_revocation.py simulate-incident
PYTHONPATH=src python scripts/check_evidence_revocation.py verify-ledger
PYTHONPATH=src python scripts/check_evidence_revocation.py audit-readiness
```

Exit codes are `0` for controls ready, `2` for ledger or propagation
violations, `3` for a valid active incident that blocks release, and `4` for
usage errors. Passing the gate proves revocation propagation and incident
response controls only. It is not Precision, Recall, human validation, or an
official score.
