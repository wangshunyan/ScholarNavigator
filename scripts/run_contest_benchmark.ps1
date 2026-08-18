[CmdletBinding()]
param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",

    [ValidateSet("local", "hybrid", "hybrid_deep_rrf", "network_hybrid")]
    [string]$Configuration = "hybrid",

    [string]$RunId = "",

    [int]$Offset = 0,

    [int]$Limit = 0,

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

if ($Limit -le 0) {
    $Limit = if ($Mode -eq "full") { 1000 } else { 5 }
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
        if ($Configuration -eq "hybrid_deep_rrf") {
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
    "--query-planning-policy", "current_rules",
    "--judgement-policy", "current_rules",
    "--ranking-policy", $(if ($Configuration -eq "hybrid_deep_rrf") { "rrf_fusion" } else { "current_rules" }),
    "--max-workers", "1",
    "--sources", $(
        if ($Configuration -in @("hybrid", "hybrid_deep_rrf")) {
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

if ($Configuration -eq "hybrid_deep_rrf") {
    $arguments += @(
        "--max-candidate-papers",
        "300"
    )
}

if ($Configuration -in @("hybrid", "hybrid_deep_rrf")) {
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

if ($Resume) {
    $arguments += "--resume"
}

Push-Location $projectRoot
try {
    $env:OPENBLAS_NUM_THREADS = "1"
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
