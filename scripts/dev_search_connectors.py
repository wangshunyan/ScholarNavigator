"""Manual development script for real connector calls.

This script is intentionally not used by tests.
"""

from __future__ import annotations

import argparse

from scholar_agent.connectors import search_arxiv, search_openalex


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real connector searches.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum results per connector")
    args = parser.parse_args()

    for source_name, search in (
        ("OpenAlex", search_openalex),
        ("arXiv", search_arxiv),
    ):
        print(f"\n## {source_name}")
        papers = search(args.query, limit=args.limit)
        if not papers:
            print("No results or connector unavailable.")
            continue
        for index, paper in enumerate(papers, start=1):
            print(f"{index}. {paper.title} ({paper.year or 'unknown year'})")
            print(f"   authors: {', '.join(paper.authors) or 'unknown'}")
            print(f"   venue: {paper.venue or 'unknown'}")
            print(f"   sources: {', '.join(paper.sources)}")
            print(f"   ids: {paper.identifiers.model_dump(exclude_none=True)}")


if __name__ == "__main__":
    main()
