# Manual Smoke Qrels Template

This fixture is a small editable template for manually judging five real-search
smoke queries. Copy `qrels.template.jsonl` to a working qrels file, then fill
`relevant_papers` with identifiers for papers judged relevant.

Each relevant paper can include:

```json
{
  "title": "Paper title",
  "year": 2025,
  "doi": "10.xxxx/example",
  "arxiv_id": "2501.00001",
  "semantic_scholar_id": "..."
}
```

Run the batch search:

```bash
PYTHONPATH=src python scripts/run_search_batch.py \
  --input datasets/eval_fixtures/manual_smoke/queries.jsonl \
  --output outputs/manual_smoke/result.jsonl \
  --top-k 5 \
  --run-profile fast \
  --current-year 2026 \
  --sources arxiv,semantic_scholar
```

Summarize the batch output:

```bash
PYTHONPATH=src python scripts/summarize_search_batch.py \
  --input outputs/manual_smoke/result.jsonl \
  --output outputs/manual_smoke/summary.md
```

Evaluate against the manually filled qrels:

```bash
PYTHONPATH=src python scripts/evaluate_search_batch.py \
  --batch-results outputs/manual_smoke/result.jsonl \
  --gold datasets/eval_fixtures/manual_smoke/qrels.filled.jsonl \
  --output outputs/manual_smoke/eval.json \
  --k 1 \
  --k 5 \
  --include-partial
```
