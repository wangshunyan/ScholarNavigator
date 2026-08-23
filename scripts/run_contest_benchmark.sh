#!/usr/bin/env bash
set -euo pipefail

MODE="smoke"
CONFIGURATION="hybrid"
RUN_ID=""
OFFSET="0"
LIMIT="0"
RESUME="0"
RERANKER_MODEL="datasets/semantic/models/Qwen3-Reranker-0.6B"
RERANKER_DEVICE="auto"
MAX_WORKERS="1"
QUALITY_EVIDENCE_LEDGER=""
QUALITY_EVIDENCE_CANDIDATE_IDENTIFIERS=""
QUALITY_EVIDENCE_CANDIDATE_REPORT=""

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
    --reranker-model)
      RERANKER_MODEL="$2"
      shift 2
      ;;
    --reranker-device)
      RERANKER_DEVICE="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --quality-evidence-ledger)
      QUALITY_EVIDENCE_LEDGER="$2"
      shift 2
      ;;
    --quality-evidence-candidate-identifiers)
      QUALITY_EVIDENCE_CANDIDATE_IDENTIFIERS="$2"
      shift 2
      ;;
    --quality-evidence-candidate-report)
      QUALITY_EVIDENCE_CANDIDATE_REPORT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$MODE" in
  smoke|qualification|full) ;;
  *)
    echo "mode must be smoke or full" >&2
    exit 2
    ;;
esac

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || [[ "$MAX_WORKERS" -gt 32 ]]; then
  echo "max-workers must be an integer from 1 to 32" >&2
  exit 2
fi

if ! [[ "$RERANKER_DEVICE" =~ ^(auto|cpu|cuda(:[0-9]+)?)$ ]]; then
  echo "reranker-device must be auto, cpu, cuda, or cuda:<index>" >&2
  exit 2
fi

case "$CONFIGURATION" in
  local|hybrid|hybrid_deep_rrf|network_hybrid|rules|dense|reranker|dense_reranker_soft|dense_reranker_rrf_soft|dense_reranker_quality|dense_reranker_llm|dense_reranker_llm_feedback) ;;
  *)
    echo "configuration must be a supported contest configuration" >&2
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
  elif [[ "$MODE" == "qualification" ]]; then
    LIMIT="200"
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
if [[ "$MODE" == "qualification" || "$CONFIGURATION" == "hybrid_deep_rrf" || "$CONFIGURATION" == "dense" || "$CONFIGURATION" == "reranker" || "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" || "$CONFIGURATION" == "dense_reranker_quality" || "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  RUN_PROFILE="high_recall"
fi

SOURCES="local_bm25"
if [[ "$CONFIGURATION" == "hybrid" || "$CONFIGURATION" == "hybrid_deep_rrf" || "$CONFIGURATION" == "dense" || "$CONFIGURATION" == "reranker" || "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" || "$CONFIGURATION" == "dense_reranker_quality" || "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  SOURCES="local_hybrid"
elif [[ "$CONFIGURATION" == "network_hybrid" ]]; then
  SOURCES="local_bm25,arxiv"
fi

RANKING_POLICY="current_rules"
if [[ "$CONFIGURATION" == "hybrid_deep_rrf" ]]; then
  RANKING_POLICY="rrf_fusion"
elif [[ "$CONFIGURATION" == "dense_reranker_quality" ]]; then
  RANKING_POLICY="quality_soft_v1"
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
  "--query-planning-policy" "$([[ "$CONFIGURATION" == "dense_reranker_llm" ]] && echo llm_semantic || echo current_rules)"
  "--judgement-policy" "current_rules"
  "--ranking-policy" "$RANKING_POLICY"
  "--max-workers" "$MAX_WORKERS"
  "--sources" "$SOURCES"
  "--local-bm25-corpus" "datasets/local_bm25/pasa_papers.jsonl"
  "--local-bm25-document-id-identity" "arxiv_id"
  "--local-bm25-arxiv-id-field" "arxiv_id"
  "--local-bm25-doi-field" "doi"
)

if [[ "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" ]]; then
  ARGS+=("--judgement-config" "benchmark/judgement_soft_current_rules_v1.json")
fi

if [[ "$MODE" == "qualification" || "$CONFIGURATION" == "hybrid_deep_rrf" || "$CONFIGURATION" == "dense" || "$CONFIGURATION" == "reranker" || "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" || "$CONFIGURATION" == "dense_reranker_quality" || "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  ARGS+=("--max-candidate-papers" "300")
fi

if [[ "$CONFIGURATION" == "hybrid" || "$CONFIGURATION" == "hybrid_deep_rrf" || "$CONFIGURATION" == "dense" || "$CONFIGURATION" == "reranker" || "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" || "$CONFIGURATION" == "dense_reranker_quality" || "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
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

if [[ "$CONFIGURATION" == "reranker" || "$CONFIGURATION" == "dense_reranker_soft" || "$CONFIGURATION" == "dense_reranker_rrf_soft" || "$CONFIGURATION" == "dense_reranker_quality" || "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  ARGS+=(
    "--local-hybrid-reranker-model" "$RERANKER_MODEL"
    "--local-hybrid-reranker-candidate-limit" "120"
    "--local-hybrid-reranker-batch-size" "8"
    "--local-hybrid-reranker-device" "$RERANKER_DEVICE"
  )
fi

if [[ "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  ARGS+=("--enable-query-evolution" "--query-evolution-policy" "llm_feedback")
fi

if [[ "$CONFIGURATION" == "dense_reranker_llm" || "$CONFIGURATION" == "dense_reranker_llm_feedback" ]]; then
  # SearchBudget is instantiated for each benchmark query. One therefore
  # enforces the competition contract per query, rather than capping the
  # entire 1000-query run at an arbitrary total.
  ARGS+=("--max-llm-calls" "1" "--max-search-rounds" "3")
fi

if [[ -n "$QUALITY_EVIDENCE_LEDGER" ]]; then
  if [[ "$CONFIGURATION" != "dense_reranker_quality" ]]; then
    echo "quality evidence requires dense_reranker_quality" >&2
    exit 2
  fi
  if [[ -z "$QUALITY_EVIDENCE_CANDIDATE_IDENTIFIERS" || -z "$QUALITY_EVIDENCE_CANDIDATE_REPORT" ]]; then
    echo "quality evidence requires candidate identifiers and report" >&2
    exit 2
  fi
  ARGS+=(
    "--quality-evidence-ledger" "$QUALITY_EVIDENCE_LEDGER"
    "--quality-evidence-candidate-identifiers" "$QUALITY_EVIDENCE_CANDIDATE_IDENTIFIERS"
    "--quality-evidence-candidate-report" "$QUALITY_EVIDENCE_CANDIDATE_REPORT"
  )
elif [[ -n "$QUALITY_EVIDENCE_CANDIDATE_IDENTIFIERS" || -n "$QUALITY_EVIDENCE_CANDIDATE_REPORT" ]]; then
  echo "quality evidence candidate binding requires ledger" >&2
  exit 2
fi

if [[ "$RESUME" == "1" ]]; then
  ARGS+=("--resume")
fi

cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS=1
LOG_RELATIVE="outputs/run_logs/${RUN_ID}.log"
LOG_PATH="$PROJECT_ROOT/$LOG_RELATIVE"
mkdir -p "$(dirname "$LOG_PATH")"
export SCHOLARNAVIGATOR_RUN_LOG_PATH="$LOG_RELATIVE"
set +e
"$PYTHON" "${ARGS[@]}" 2>&1 | tee -a "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e
exit "$STATUS"
