#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
CONFIGURATION="hybrid"
RUN_ID=""
OFFSET="0"
LIMIT="0"
RESUME="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --configuration)
      CONFIGURATION="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  smoke|full) ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 2
    ;;
esac

case "$CONFIGURATION" in
  local|hybrid|hybrid_deep_rrf|network_hybrid) ;;
  *)
    echo "configuration must be local, hybrid, hybrid_deep_rrf, or network_hybrid" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python virtual environment not found: $PYTHON" >&2
  exit 1
fi

if [[ "$LIMIT" -le 0 ]]; then
  if [[ "$MODE" == "full" ]]; then
    LIMIT="1000"
  else
    LIMIT="5"
  fi
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="contest_${CONFIGURATION}_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

RUN_PROFILE="evaluation"
if [[ "$MODE" != "full" ]]; then
  RUN_PROFILE="fast"
fi
if [[ "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  RUN_PROFILE="high_recall"
fi

SOURCES="local_bm25"
if [[ "$CONFIGURATION" == "hybrid" || "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  SOURCES="local_hybrid"
elif [[ "$CONFIGURATION" == "network_hybrid" ]]; then
  SOURCES="local_bm25,arxiv"
fi

RANKING_POLICY="current_rules"
if [[ "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  RANKING_POLICY="rrf_fusion"
fi

ARGS=(
  "scripts/run_benchmark.py"
  "--dataset" "auto_scholar_query"
  "--dataset-split" "test"
  "--run-id" "$RUN_ID"
  "--offset" "$OFFSET"
  "--limit" "$LIMIT"
  "--run-profile" "$RUN_PROFILE"
  "--top-k" "20"
  "--diagnostics"
  "--resource-ledger"
  "--query-adapter-policy" "adaptive"
  "--query-planning-policy" "current_rules"
  "--judgement-policy" "current_rules"
  "--ranking-policy" "$RANKING_POLICY"
  "--max-workers" "1"
  "--sources" "$SOURCES"
  "--local-bm25-corpus" "datasets/local_bm25/pasa_papers.jsonl"
  "--local-bm25-document-id-identity" "arxiv_id"
  "--local-bm25-arxiv-id-field" "arxiv_id"
  "--local-bm25-doi-field" "doi"
)

if [[ "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  ARGS+=("--max-candidate-papers" "300")
fi

if [[ "$CONFIGURATION" == "hybrid" || "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  BM25_LIMIT="60"
  SEMANTIC_LIMIT="60"
  if [[ "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
    BM25_LIMIT="120"
    SEMANTIC_LIMIT="120"
  fi
  ARGS+=(
    "--local-hybrid-semantic-corpus"
    "datasets/semantic/pasa_papers_with_abstracts.jsonl"
    "--local-hybrid-index-dir"
    "outputs/benchmark_cache/local_hybrid"
    "--local-hybrid-model"
    "datasets/semantic/models/models/AI-ModelScope--bge-small-en-v1.5/snapshots/master"
    "--local-hybrid-bm25-candidate-limit"
    "$BM25_LIMIT"
    "--local-hybrid-semantic-candidate-limit"
    "$SEMANTIC_LIMIT"
    "--local-hybrid-rrf-k"
    "60"
  )
fi

if [[ "$RESUME" == "1" ]]; then
  ARGS+=("--resume")
fi

cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS=1
"$PYTHON" "${ARGS[@]}"
