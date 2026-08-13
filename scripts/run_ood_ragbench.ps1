# Out-of-distribution evaluation: the RAGTruth-trained C1 checkpoint on RAGBench.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_ood_ragbench.ps1
#
# Inference only. RAGBench is never trained on and never used to select
# anything -- it is the held-out cross-domain testbed and touching it for
# anything else would destroy the only claim this table makes.
#
# Two steps:
#   1. Download the 12 test splits and convert them to processed records.
#      Network is needed once; huggingface_hub caches afterwards. ~54 MB of
#      parquet in, ~80 MB of JSONL out under data\processed\ragbench (gitignored).
#   2. Run the checkpoint over each subset and write the OOD table.
#
# READ THIS BEFORE READING THE TABLE. RAGBench's labels are written by an LLM
# (gpt-4-turbo on 10,742 records, gpt-4o on 1,059), not by humans, and they mark
# whole sentences where RAGTruth marks phrases. Only example-level F1 is
# comparable across the two corpora, and evaluate_ood prints only that. The
# positive rate is 14.2% here against RAGTruth's 43.1%, so part of any F1 drop
# is the base rate moving, not the detector failing -- the table prints the
# per-subset positive rate next to the F1 for exactly that reason.
#
# Batch size defaults to 4, not evaluate_c1's 8. cuad has a median context of
# 26,125 characters and techqa 17,640, so these batches fill max_length in a way
# the RAGTruth run never did, and 8 x 3072 does not fit comfortably in 6GB.
# Raise it on a bigger card.

param(
    [string]$Checkpoint = "results\c1\modernbert-base\best",
    [int]$BatchSize = 4,
    [int]$MaxLength = 3072,
    [switch]$SkipDownload,
    [string[]]$Subsets
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "no virtualenv at $python" }
if (-not (Test-Path $Checkpoint)) {
    throw "checkpoint not found at $Checkpoint -- unpack the Kaggle artifacts first"
}

$dataDir = "data\processed\ragbench"

if (-not $SkipDownload) {
    Write-Host "`n== RAGBench: download and convert the 12 test splits ==" -ForegroundColor Cyan
    $buildArgs = @("-u", "-m", "src.c1_detector.ragbench", "--out-dir", $dataDir)
    if ($Subsets) { $buildArgs += @("--subsets") + $Subsets }
    & $python @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "RAGBench conversion failed" }
}

Write-Host "`n== C1 on RAGBench, zero-shot ==" -ForegroundColor Cyan
$evalArgs = @(
    "-u", "-m", "src.c1_detector.evaluate_ood",
    "--checkpoint", $Checkpoint,
    "--data-dir", $dataDir,
    "--reference", "results\c1\test\metrics.json",
    "--out-dir", "results\ood\ragbench",
    "--batch-size", $BatchSize,
    "--max-length", $MaxLength
)
if ($Subsets) { $evalArgs += @("--subsets") + $Subsets }
& $python @evalArgs
if ($LASTEXITCODE -ne 0) { throw "OOD evaluation failed" }

Write-Host "`n== done ==" -ForegroundColor Green
Write-Host "results\ood\ragbench\ood_table.txt     the table for the paper"
Write-Host "results\ood\ragbench\ood_metrics.json  full metrics, all levels, for the record"
Write-Host ""
Write-Host "Report example-level F1 only, and report it beside the positive rate." -ForegroundColor Yellow
Write-Host "The span and token numbers in the JSON compare a phrase detector" -ForegroundColor Yellow
Write-Host "against a sentence annotation and do not mean what they look like." -ForegroundColor Yellow
