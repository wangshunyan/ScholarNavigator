# Offline wheelhouse intake

`offline_wheelhouse_intake_v1` is the supply-chain boundary between the
version contract in `python_dependency_lock_v1` and any future offline Python
installation. It never downloads or manufactures a missing dependency.

The intake manifest binds each required normalized distribution name and exact
version to one wheel filename, byte size, SHA-256, Python/ABI/platform tags,
`Requires-Dist`, entry points, and the license statement present in wheel
metadata. Artifact provenance remains `unknown` unless a separately attested
source type is supplied; an installed distribution is not accepted as a wheel.

Verification rejects sdists, missing or extra distributions, duplicate and
confusable names, incompatible tags, filename/metadata disagreement, malformed
or incomplete `RECORD`, unsafe ZIP paths, links, duplicate members, excessive
expansion, invalid entry points, and a dependency closure that differs from the
locked manifest and SBOM. The generated installation plan uses exact versions
and `--require-hashes`, and contains no absolute path or download URL.

The default read-only commands are:

```bash
PYTHONPATH=src python scripts/check_offline_wheelhouse.py prepare-manifest
PYTHONPATH=src python scripts/check_offline_wheelhouse.py verify
PYTHONPATH=src python scripts/check_offline_wheelhouse.py install-test
PYTHONPATH=src python scripts/check_offline_wheelhouse.py audit-release
```

Exit `0` means the complete real wheelhouse passed intake and two isolated
offline installations. Exit `2` is an artifact or supply-chain violation, exit
`3` means required wheels are unavailable, and exit `4` is usage error. The
synthetic test uses temporary, locally constructed micro-wheels to exercise the
same verifier and installer. It is engineering evidence only and can never
qualify the 23 real locked dependencies.
