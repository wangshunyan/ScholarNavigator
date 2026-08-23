#!/usr/bin/env python3
"""Audit local source/GitHub synchronization without reading secrets.

This is deliberately read-only.  It reports the local commit, the configured
``origin/main`` commit when available, dirty files, and tracked files that
must never be published.  Server experiments are not contacted; their
consistency is established separately from an imported evidence manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FORBIDDEN_TRACKED = (".env", "outputs/", "datasets/semantic/", "*.pem", "*.key")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git_failed:{' '.join(args)}")
    return result.stdout.strip()


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    head = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = [line for line in _git(root, "status", "--porcelain=v1").splitlines() if line]
    try:
        origin = _git(root, "rev-parse", "origin/main")
    except RuntimeError:
        origin = None
    tracked = _git(root, "ls-files").splitlines()
    forbidden = [
        path for path in tracked
        if path == ".env"
        or path.startswith("outputs/")
        or path.startswith("datasets/semantic/")
        or path.lower().endswith((".pem", ".key"))
    ]
    return {
        "status": "ready" if not status and not forbidden and (origin is None or head == origin) else "review",
        "branch": branch,
        "head": head,
        "origin_main": origin,
        "github_in_sync": origin is not None and head == origin,
        "working_tree_clean": not status,
        "dirty_paths": status,
        "forbidden_tracked_paths": forbidden,
        "server_contacted": False,
        "notes": [
            "Server experiments require a redacted evidence bundle and are never inferred from Git status.",
            "Do not commit .env, outputs/, model/index caches, raw server runs, or SSH credentials.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.root), ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
