# Manual Extended Evaluation Skeleton

`cases.jsonl` contains 10 complex academic-search cases for extending the
manual evaluation set. Each line is a JSON object with:

- `case_id`: stable case identifier.
- `query`: user-facing academic search query.
- `intent`: coarse query type for analysis.
- `expected_gold_slots`: paper types or representative directions that qrels
  should cover.
- `notes`: guidance for future manual judging.

This directory does not contain official qrels. Future reviewers should create
a separate filled qrels JSONL file by manually adding verified relevant papers
with real identifiers such as title, year, DOI, arXiv ID, Semantic Scholar ID,
or PubMed ID.

Current files are for extended evaluation design only and should not be treated
as formal scoring qrels.
