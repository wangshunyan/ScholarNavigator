# Python dependency lock v1

`python_dependency_lock_v1` is an offline release-engineering gate. It does not
run retrieval or evaluation and does not establish an official score.

The authoritative direct declarations are split by purpose:

- `requirements.txt` contains exact runtime requirements only;
- `requirements-dev.txt` contains exact test/development requirements only;
- `requirements-runtime.lock` and `requirements-dev.lock` contain the complete
  environment-specific closures reconstructed from installed distribution
  metadata;
- `benchmark/python_dependency_lock_v1_manifest.json` records normalized names,
  versions, markers, dependency edges, metadata hashes, declared licenses and
  local offline artifact availability.

Versions are accepted only from exact repository declarations and currently
verifiable installed `METADATA`. The gate does not download packages or infer a
missing version. Marker evaluation is bound to the environment recorded in
`benchmark/python_dependency_lock_v1_protocol.json`; another interpreter or
platform must create a separately reviewed lock.

## Commands and exit codes

```bash
PYTHONPATH=src python scripts/check_python_dependency_lock.py generate
PYTHONPATH=src python scripts/check_python_dependency_lock.py verify
PYTHONPATH=src python scripts/check_python_dependency_lock.py offline-install
PYTHONPATH=src python scripts/check_python_dependency_lock.py audit-release
```

- `0`: the exact lock and two isolated `pip --no-index` installations passed;
- `2`: declaration, closure, wheel metadata or release integration violated the
  contract;
- `3`: versions are verified but a local installation artifact or other
  required offline input is unavailable;
- `4`: invalid command usage.

The offline installer never downloads dependencies. It may use only matching
wheel files already present in the local pip cache, and records only wheel
filename, version, size and SHA-256—never the cache's absolute path. When the
wheelhouse is incomplete it returns the full missing package list before
creating either venv.

The deterministic application wheel exposes only the exact runtime direct
requirements in `Requires-Dist`; `pytest`, `httpx`, and their development-only
closure are prohibited from runtime metadata. A release remains unqualified
until both isolated venv installations, imports, CLI help, minimal FastAPI app
load and uninstall residue checks pass. This engineering gate does not remove
the Full1000, human Precision, or official scorer blockers.
