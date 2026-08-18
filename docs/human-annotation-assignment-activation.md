# Human annotation assignment activation

`human_annotation_assignment_activation_v1` binds three already-qualified
opaque principals to the frozen 471-item delivery contract. It is an
assignment and blinding control, not a source of labels, Precision, or an
official score.

## State machine and bindings

Each role follows `prepared → assigned → issued → acknowledged →
locked_for_submission`. `revoked` and `invalid` are terminal alternatives.
Every append-only event binds the role, opaque principal, qualification hash,
one-time assignment challenge, package hash, protocol hash, source commit and
the previous event hash. A manual boolean cannot advance the chain.

The two annotators receive the existing independent A/B alias packages. Each
contains exactly 471 items and the alias sets are disjoint. The adjudicator
receives only the frozen rubric and a contract for a future disagreement view;
the adjudicator does not receive either annotator package, original decisions,
or the operator mapping. The operator-only alias mapping is never copied into
an issued bundle.

## Offline bundle and receipt

Bundles are deterministic ZIP files with fixed member order, timestamps and
permissions. A standard-library verifier runs under `python -I -S`. The
manifest records file sizes and SHA-256 values, but the hashes authenticate
content only—not the natural person behind an opaque principal.

An acknowledgement receipt is bound one-to-one to the principal, role,
challenge, bundle, protocol and commit. Duplicate claims, package swapping,
coordinator claims, cross-role reuse, revoked qualifications and post-issue
semantic drift fail closed. Label intake remains disabled until all three
receipts have been verified and all role chains reach
`locked_for_submission`. Semantic changes require revocation and fresh
issuance; old labels cannot be migrated.

```bash
PYTHONPATH=src python scripts/check_human_annotation_assignment.py prepare \
  --output /tmp/human-assignment-prepared.json
PYTHONPATH=src python scripts/check_human_annotation_assignment.py simulate-matrix
PYTHONPATH=src python scripts/check_human_annotation_assignment.py audit-readiness
```

Exit codes are `0=assignment_chain_ready`,
`2=assignment_or_blinding_violation`,
`3=not_ready_missing_real_qualified_principals`, and `4=usage_error`.

The tracked readiness audit returns exit 3. Real qualifications and real
acknowledgements for annotator-A, annotator-B and the adjudicator are absent.
No real package has been issued, no label has been created, and the human
Precision blocker remains active.
