# Local C1 smoke test on the RTX 3050. Catches bugs, produces no reportable number.
#
#   powershell -ExecutionPolicy Bypass -File scripts\smoke_c1.ps1
#
# 500 stratified examples for one epoch at max_length 1024. The only question it
# answers is "does the pipeline run end to end without crashing" -- 500 examples
# cannot train a detector, so nothing it prints goes in the report.
#
# Output is tee'd to a log because a long run can time out at the tool layer
# while still succeeding. Read the log, not the exit of the terminal.
#
#   -Tiny     40 examples, for checking a code change in about a minute
#   -Cpu      force CPU, to isolate a CUDA problem from a logic problem

param(
    [switch]$Tiny,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "no virtualenv at $python -- run scripts\setup_env.ps1 first"
}

$trainFile = Join-Path $repo "data\processed\ragtruth_train.jsonl"
if (-not (Test-Path $trainFile)) {
    throw "no $trainFile -- run scripts\prepare_ragtruth.ps1 first"
}

Write-Host "`n== environment ==" -ForegroundColor Cyan
& $python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); print('transformers', transformers.__version__); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if ($LASTEXITCODE -ne 0) { throw "environment check failed" }

$logDir = Join-Path $repo "results\c1"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "smoke.log"

$trainArgs = @("-m", "src.c1_detector.train_c1", "--config", "configs\c1_smoke.yaml")
if ($Tiny) { $trainArgs += @("--limit", "40") }
if ($Cpu) { $trainArgs += @("--device", "cpu") }

Write-Host "`n== training ==" -ForegroundColor Cyan
Write-Host "logging to $log"
& $python @trainArgs 2>&1 | Tee-Object -FilePath $log
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Host "`nsmoke test FAILED (exit $exit). The log is at $log" -ForegroundColor Red
    Write-Host "If it is a CUDA out-of-memory error, drop train.batch_size to 1 in" -ForegroundColor Yellow
    Write-Host "configs\c1_smoke.yaml, or set model.gradient_checkpointing: true." -ForegroundColor Yellow
    exit $exit
}

Write-Host "`n== smoke test passed ==" -ForegroundColor Green
Write-Host "The F1 numbers above are meaningless -- 500 examples, one epoch." -ForegroundColor Yellow
Write-Host "What matters:" -ForegroundColor Yellow
Write-Host "  - the loss moved" -ForegroundColor Yellow
Write-Host "  - all three tasks appeared in the per-task breakdown" -ForegroundColor Yellow
Write-Host "  - 'answers truncated' was 0" -ForegroundColor Yellow
Write-Host "  - in the diagnostics block, mean P(hal) on gold-POSITIVE tokens is" -ForegroundColor Yellow
Write-Host "    clearly higher than on gold-negative ones" -ForegroundColor Yellow
Write-Host ""
Write-Host "'pred spans 0' at argmax is EXPECTED here and is not a failure. On 400" -ForegroundColor Yellow
Write-Host "training examples the model collapses to the majority class, so every" -ForegroundColor Yellow
Write-Host "probability sits under 0.5. The threshold sweep is what shows whether the" -ForegroundColor Yellow
Write-Host "span decoder works and whether the two classes separate at all. If the two" -ForegroundColor Yellow
Write-Host "means in that block are equal, something IS wrong." -ForegroundColor Red
Write-Host ""
Write-Host "Next: upload data\processed\*.jsonl as a private Kaggle Dataset, then run"
Write-Host "notebooks\c1_kaggle_train.ipynb with Save and Run All (Commit)."
