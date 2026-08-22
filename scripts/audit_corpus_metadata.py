from __future__ import annotations

import argparse
import json
from pathlib import Path

from scholar_agent.evaluation.corpus_metadata_audit import audit_jsonl_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a local paper JSONL corpus")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--identity-field", default="arxiv_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_jsonl_corpus(args.corpus, identity_field=args.identity_field)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
