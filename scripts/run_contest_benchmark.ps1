[CmdletBinding()]
param(
    [ValidateSet("smoke", "qualification", "full")]
    [string]$Mode = "smoke",

    [ValidateSet("local", "hybrid", "hybrid_deep_rrf", "network_hybrid", "rules", "dense", "reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")]
    [string]$Configuration = "hybrid",

    [string]$RunId = "",

    [int]$Offset = 0,

    [int]$Limit = 0,

    [switch]$Resume,

    [ValidateRange(1, 32)]
    [int]$MaxWorkers = 1,

    [string]$RerankerModel = "datasets\semantic\models\Qwen3-Reranker-0.6B",

    [ValidatePattern("^(auto|cpu|cuda(:\d+)?)$")]
    [string]$RerankerDevice = "auto"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

if ($Limit -le 0) {
$Limit = if ($Mode -eq "full") { 1000 } elseif ($Mode -eq "qualification") { 200 } else { 5 }
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunId = "contest_${Configuration}_${Mode}_${timestamp}"
}

$arguments = @(
    "scripts\run_benchmark.py",
    "--dataset", "auto_scholar_query",
    "--dataset-split", "test",
    "--run-id", $RunId,
    "--offset", $Offset,
    "--limit", $Limit,
    "--run-profile", $(
        if ($Configuration -in @("hybrid_deep_rrf", "dense", "reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")) {
            "high_recall"
        } elseif ($Mode -eq "full") {
            "evaluation"
        } else {
            "fast"
        }
    ),
    "--top-k", "20",
    "--diagnostics",
    "--resource-ledger",
    "--query-adapter-policy", "adaptive",
    "--query-planning-policy", $(if ($Configuration -eq "dense_reranker_llm") { "llm_semantic" } else { "current_rules" }),
    "--judgement-policy", "current_rules",
    "--ranking-policy", $(if ($Configuration -eq "hybrid_deep_rrf") { "rrf_fusion" } else { "current_rules" }),
    "--max-workers", $MaxWorkers,
    "--sources", $(
        if ($Configuration -in @("hybrid", "hybrid_deep_rrf", "dense", "reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")) {
            "local_hybrid"
        } elseif ($Configuration -eq "network_hybrid") {
            "local_bm25,arxiv"
        } else {
            "local_bm25"
        }
    ),
    "--local-bm25-corpus", "datasets\local_bm25\pasa_papers.jsonl",
    "--local-bm25-document-id-identity", "arxiv_id",
    "--local-bm25-arxiv-id-field", "arxiv_id",
    "--local-bm25-doi-field", "doi"
)

if ($Configuration -in @("hybrid_deep_rrf", "dense", "reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")) {
    $arguments += @(
        "--max-candidate-papers",
        "300"
    )
}

if ($Configuration -in @("hybrid", "hybrid_deep_rrf", "dense", "reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")) {
    $arguments += @(
        "--local-hybrid-semantic-corpus",
        "datasets\semantic\pasa_papers_with_abstracts.jsonl",
        "--local-hybrid-index-dir",
        "outputs\benchmark_cache\local_hybrid",
        "--local-hybrid-model",
        "datasets\semantic\models\models\AI-ModelScope--bge-small-en-v1.5\snapshots\master",
        "--local-hybrid-bm25-candidate-limit",
        $(if ($Configuration -eq "hybrid_deep_rrf") { "120" } else { "60" }),
        "--local-hybrid-semantic-candidate-limit",
        $(if ($Configuration -eq "hybrid_deep_rrf") { "120" } else { "60" }),
        "--local-hybrid-rrf-k",
        "60"
    )
}

if ($Configuration -eq "dense_reranker_soft") {
    $arguments += @(
        "--judgement-config",
        "benchmark\judgement_soft_current_rules_v1.json"
    )
}

if ($Configuration -in @("reranker", "dense_reranker_soft", "dense_reranker_llm", "dense_reranker_llm_feedback")) {
    $arguments += @(
        "--local-hybrid-reranker-model", $RerankerModel,
        "--local-hybrid-reranker-candidate-limit", "120",
        "--local-hybrid-reranker-batch-size", "8",
        "--local-hybrid-reranker-device", $RerankerDevice
    )
}

if ($Configuration -eq "dense_reranker_llm_feedback") {
    $arguments += @(
        "--enable-query-evolution",
        "--query-evolution-policy", "llm_feedback"
    )
}

if ($Configuration -in @("dense_reranker_llm", "dense_reranker_llm_feedback")) {
    $arguments += @(
        "--max-llm-calls", "1",
        "--max-search-rounds", "3"
    )
}

if ($Resume) {
    $arguments += "--resume"
}

Push-Location $projectRoot
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    $logRelative = "outputs/run_logs/$RunId.log"
    $logPath = Join-Path $projectRoot $logRelative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
    $env:SCHOLARNAVIGATOR_RUN_LOG_PATH = $logRelative
    & $python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
