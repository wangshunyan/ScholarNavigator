# Independent standalone auditor bundle

`standalone_auditor_bundle_v1` packages the shareable part of the tracked
validation-readiness evidence as deterministic data. It is an engineering
integrity check, not evidence of retrieval quality, human Precision, or an
official score.

The ZIP contains canonical JSON summaries for claims, blockers, freshness,
default-policy state, evidence references, and protocol dependencies, plus one
`verify.py`. Bottom-level internal artifacts are not copied into the archive;
their stable hashes are marked `externally_unverifiable_reference`. This lets
an external reviewer check the publication's closed hash chain without
claiming that a hash authenticates the publisher or independently proves the
unpublished artifact's contents.

The verifier uses only the Python standard library. A reviewer receives a
trusted copy of the repository verifier separately from the archive under
review, copies that trusted file into a checkout-free directory, and runs:

```bash
python -I -S verify.py verify standalone-auditor.zip
```

It reads only the named archive, does not import SPAR modules, execute archive
members, use the network, or launch subprocesses. The ZIP's `verify.py` is
checked only as inert data against the trusted verifier hash; it is never
extracted and executed during verification. It rejects extra, missing,
duplicate, linked, absolute, traversing, Unicode-colliding, oversized, highly
compressed, non-UTF-8, non-canonical, duplicate-key, and non-finite JSON
members. This separation avoids treating code supplied by the object under
review as the verifier's trust root.

New archives also carry a hash-bound `revocation.json` summary. The trusted
verifier rejects legacy archives that do not declare revocation state and
rejects any active incident. A hash-intact old archive therefore cannot bypass
the current revocation boundary.

Repository operators build and compare temporary archives with:

```bash
python scripts/check_standalone_auditor_bundle.py build \
  --output /tmp/standalone-auditor.zip
python scripts/check_standalone_auditor_bundle.py verify \
  /tmp/standalone-auditor.zip
python scripts/check_standalone_auditor_bundle.py compare \
  /tmp/standalone-auditor-a.zip /tmp/standalone-auditor-b.zip
python scripts/check_standalone_auditor_bundle.py audit-readiness
```

Exit codes are `0=verified_with_declared_blockers`,
`2=integrity_or_claim_violation`,
`3=not_ready_missing_shareable_evidence`, and `4=usage_error`. Generated ZIPs
are temporary release products and are intentionally not tracked.

The current archive preserves `formal_validation_complete=false` and all three
formal blockers: Full1000 completion, real independent human Precision, and an
exact official scorer/schema.

New archives also include a redacted `preregistration.json`. It exposes only
the sealed state, protocol/seal digests, and the same three external blockers.
The trusted repository verifier checks the tracked preregistration seal and
its registered dependencies before building; the archive verifier then checks
the redacted member without receiving queries, labels, or scorer definitions.
