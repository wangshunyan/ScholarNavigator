# ScholarNavigator Frontend

Next.js + TypeScript + Tailwind CSS prototype for ScholarNavigator.

## Requirements

- Node.js 20+
- npm 10+
- FastAPI backend running at `http://localhost:8000`

## Install

```bash
cd frontend
npm install
```

## Configure Backend URL

The frontend defaults to:

```text
http://localhost:8000
```

Override it with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

## Start Development Server

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

If port `3000` is already in use, Next.js may switch to another port such as
`3001`. The backend CORS defaults allow common local frontend ports:

- `http://localhost:3000`
- `http://127.0.0.1:3000`
- `http://localhost:3001`
- `http://127.0.0.1:3001`
- `http://localhost:5173`
- `http://127.0.0.1:5173`

To add a custom frontend origin, start the backend with:

```bash
SCHOLAR_AGENT_CORS_ORIGINS=http://localhost:4321 \
PYTHONPATH=src uvicorn scholar_agent.app.main:app --host 127.0.0.1 --port 8000
```

`SCHOLAR_AGENT_CORS_ORIGINS` is comma-separated. Values are trimmed, empty
entries are ignored, and configured origins are merged with the default
allowlist.

## Build

```bash
npm run build
```

## Lint

```bash
npm run lint
```

## Backend

Start the FastAPI Real Search backend from the repository root:

```bash
PYTHONPATH=src uvicorn scholar_agent.app.main:app --reload --host 127.0.0.1 --port 8000
```

Runtime config:

```bash
curl http://127.0.0.1:8000/api/v1/runtime/config
```

The backend reports a Real Search runtime. Product-facing example-search endpoints
and example UI mode are removed; searches use the Real Search lifecycle with
OpenAlex, arXiv, and Semantic Scholar connectors.

The backend can optionally enable OpenAI-compatible LLM Query Understanding and
LLM Judgement by environment variables. The frontend never reads, stores, or
displays the LLM API key. If the backend LLM provider is disabled or unavailable,
the backend uses deterministic rule-based paths and surfaces explicit
diagnostics such as `llm_query_understanding_disabled`,
`llm_query_understanding_failed:<reason>`, `llm_judgement_disabled`, or
`llm_judgement_failed:<reason>`.

Backend-only LLM environment variables can be placed in the repository-root
`.env` file. Copy the template first:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
SCHOLAR_AGENT_LLM_PROVIDER=openai_compatible
SCHOLAR_AGENT_LLM_BASE_URL=https://api.openai.com/v1
SCHOLAR_AGENT_LLM_API_KEY=your_api_key
SCHOLAR_AGENT_LLM_MODEL=gpt-4.1-mini
SCHOLAR_AGENT_ENABLE_LLM_QUERY_UNDERSTANDING=1
SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT=1
SCHOLAR_AGENT_LLM_JUDGEMENT_BATCH_SIZE=8
SEMANTIC_SCHOLAR_API_KEY=
```

The FastAPI backend automatically loads `.env` on startup. The frontend `.env`
is not used for LLM secrets.

This LLM integration is currently limited to Query Understanding and relevance
Judgement. LLM Judgement only evaluates provided candidate-paper metadata; it
does not read full-text PDFs, generate new papers, or introduce papers outside
the candidate list. Reranking and Synthesis remain rule-based.

OpenAlex, arXiv, and Semantic Scholar are implemented for Real Search, but live
calls can still be affected by external service failures such as OpenAlex `503`,
arXiv `429`, Semantic Scholar `429`, or timeouts. Those diagnostics are surfaced through Real Search events,
`missing_evidence`, and cost/source statistics when reported by the backend.

## Real Search Workflow

The workbench has one product search path:

- `Real Search`: uses the asynchronous real run lifecycle under `/api/v1/real/search/runs`, including real run creation, queued/running/succeeded/failed status polling, result fetch after completion, and real search SSE events.

`Real Search` may access real OpenAlex, arXiv, and Semantic Scholar through the backend. It still does not read, store, or display API keys in the frontend.

The Search Workbench includes a `source_preferences` selector with `Recommended`,
`arXiv`, `Semantic Scholar`, `OpenAlex`, and `All`. It defaults to `Recommended`,
which maps to `["arxiv", "semantic_scholar"]` and balances stability with
coverage. `All` maps to `["openalex", "arxiv", "semantic_scholar"]` for maximum
coverage, but OpenAlex may be less stable and can return `503`.
The recommended default is optimized for stability and low latency:
`top_k=5`, `run_profile=fast`, `source_preferences=["arxiv", "semantic_scholar"]`,
`enable_query_evolution=false`, `enable_refchain=false`, LLM Query
Understanding disabled, and LLM Judgement disabled. For higher recall, manually
enable Query Evolution, RefChain, or `All` sources. The
`enable_llm_query_understanding` toggle can improve query parsing when the
backend LLM provider is configured, but it adds latency. The
`enable_llm_judgement` toggle can improve metadata relevance judgement, but it
also adds latency. The frontend never reads or stores the LLM API key.

The backend still exposes `POST /api/v1/internal/search/preview/api-result` for
debugging the mapper path, but it is not the frontend's product search path and
does not return mock data.

If real retrieval returns no visible papers, check the `missing_evidence` diagnostics in the Results panel. Network failures such as OpenAlex 503 or arXiv timeout are surfaced there when the backend reports them.

If the SSE connection fails while status polling continues, the UI keeps the
current run status/result and records an `sse_error` event instead of clearing
the results.

`Real Search` events include the stage lifecycle events plus retrieval
observability events:

- `connector_completed`: one event per connector/source with `returned_count`,
  `latency_seconds`, `cache_hit`, and any connector `error_message`
- `warning`: backend search warnings from retrieval, query evolution, RefChain,
  synthesis, or connector diagnostics
- `cost_updated`: the final mapped cost report once the result is available

`Real Search` also supports cancelling a queued/running real search run. The UI
calls `POST /api/v1/real/search/runs/{run_id}/cancel`, stops polling, closes the
SSE connection, and keeps already received events visible. The current backend
cannot force-kill connector calls that are already executing; cancellation marks
the run as `cancelled`, ignores any later background result, and stops the
frontend from waiting for completion.

When backend retrieval cache is enabled, repeated Real Search runs in the same
backend process may report `cache_hit_count` in the cost report and show cache
hit diagnostics in `missing_evidence`.

## Synthesis Panel

`Real Search` may return an optional `synthesis` object in `SearchRunResultResponse`.
When present, the Results area renders a citation-backed synthesis panel above the
paper lists with:

- `answer_summary`
- `status`
- key findings with citation keys and confidence
- citation coverage counters
- limitations and warnings
- the first evidence-table rows

The current synthesis MVP is rule-based and grounded in ranked-paper metadata
and evidence rows. It is citation-backed, but it does not mean the system has
read full-text PDFs.

## Citation Graph Panel

The Results area also renders a Citation Graph panel when
`SearchRunResultResponse.citation_graph.nodes` or `.edges` is non-empty.

The panel shows:

- node and edge counts
- node list with `label`, `id`, and optional `rank`
- edge list with `source`, `target`, and `relation`
- an explicit empty-edge state when nodes exist but no edges were returned

The graph panel only displays structured citation graph / RefChain metadata
returned by the backend. The frontend does not infer missing citation
relationships or create graph edges on its own.

## Result Export

When a search result is loaded, the Results header shows:

- `Export JSON`
- `Export Markdown`

`Export JSON` downloads the complete current `SearchRunResultResponse` object as:

```text
scholar-navigator-result-{run_id}.json
```

`Export Markdown` downloads a readable report as:

```text
scholar-navigator-result-{run_id}.md
```

The Markdown report includes run metadata, query analysis, expanded queries,
cost report, optional synthesis, highly relevant papers, partially relevant
papers, citation graph nodes/edges, and missing evidence diagnostics.

Both export actions run entirely in the browser with `Blob` and
`URL.createObjectURL`. They use the result already present on the page, do not
trigger a new search, do not upload data to the backend, and do not access any
external service.
